"""Worker-side KV block transfers for AgentKVConnector (plain PyTorch, device agnostic).

vLLM's FlashAttention layer tensors have the logical shape
(num_blocks, kv_heads, block_size, 2 * head_dim), keys first. A CPU slot holds one block for
all layers: a bf16 copy (layers, heads, tokens, 2 * head_dim) and, if lo_bits is set, a packed
low-bit row from kvm.quant.pack_blocks. Copies and (de)quantization run on a side CUDA stream;
completion is polled with CUDA events, so the engine keeps serving other requests meanwhile.
"""

import contextlib

import torch

from kvm.quant import block_bytes, pack_blocks, unpack_blocks
from kvm.sched import LoadJob, SaveJob


def _pin(t: torch.Tensor) -> torch.Tensor:
    """Page-lock a CPU tensor in place. pin_memory=True would round the allocation up to the
    next power of two (PyTorch's caching host allocator); vLLM pins the same way."""
    err = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.nbytes, 0)
    if err.value != 0:
        raise RuntimeError(f"cudaHostRegister failed: {err}")
    return t


def _runs(slots: list[int]):
    """(offset, first slot, length) for each run of consecutive slots."""
    start = 0
    for i in range(1, len(slots) + 1):
        if i == len(slots) or slots[i] != slots[i - 1] + 1:
            yield start, slots[start], i - start
            start = i


class KVTransfer:
    def __init__(self, layers: list[torch.Tensor], n_slots: int, lo_bits: int | None,
                 chunk_blocks: int = 64, pin_memory: bool | None = None):
        first = layers[0]
        self.layers = layers
        self.device = first.device
        self.dtype = first.dtype
        _, self.heads, self.tokens, d2 = first.shape
        self.head_dim = d2 // 2
        self.lo_bits = lo_bits
        self.chunk = chunk_blocks
        pin = self.device.type == "cuda" if pin_memory is None else pin_memory
        maybe_pin = _pin if pin else (lambda t: t)
        n_layers = len(layers)
        self.bf16_pool = maybe_pin(torch.empty((n_slots, n_layers, self.heads, self.tokens, d2), dtype=self.dtype))
        self.lo_pool = None
        if lo_bits:
            row = n_layers * block_bytes(lo_bits, self.tokens, self.heads, self.head_dim)
            self.lo_pool = maybe_pin(torch.empty((n_slots, row), dtype=torch.uint8))
        self.stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
        self._sending: dict[str, torch.cuda.Event | None] = {}
        self._recving: dict[str, torch.cuda.Event | None] = {}
        self.bytes = {"d2h": 0, "h2d": 0}

    def _on_stream(self):
        if self.stream is None:
            return contextlib.nullcontext()
        # Order after everything already queued on the compute stream (the forward that
        # wrote the blocks being saved, or the last use of the blocks being overwritten).
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        return torch.cuda.stream(self.stream)

    def _event(self):
        if self.stream is None:
            return None
        ev = torch.cuda.Event()
        ev.record(self.stream)
        return ev

    def _ids(self, blocks: list[int]) -> torch.Tensor:
        ids = torch.tensor(blocks, dtype=torch.long)
        if self.device.type != "cuda":
            return ids
        # A pageable upload would synchronize the side stream on every chunk.
        return ids.pin_memory().to(self.device, non_blocking=True)

    def _gather(self, blocks: list[int]) -> torch.Tensor:
        ids = self._ids(blocks)
        return torch.stack([layer[ids] for layer in self.layers], dim=1)

    def _scatter(self, blocks: list[int], data: torch.Tensor) -> None:
        ids = self._ids(blocks)
        for i, layer in enumerate(self.layers):
            layer[ids] = data[:, i]

    def _to_host(self, pool: torch.Tensor, slots: list[int], src: torch.Tensor) -> None:
        for off, s, n in _runs(slots):
            pool[s:s + n].copy_(src[off:off + n], non_blocking=True)
        self.bytes["d2h"] += src.numel() * src.element_size()

    def _to_device(self, pool: torch.Tensor, slots: list[int]) -> torch.Tensor:
        dst = torch.empty((len(slots), *pool.shape[1:]), dtype=pool.dtype, device=self.device)
        for off, s, n in _runs(slots):
            dst[off:off + n].copy_(pool[s:s + n], non_blocking=True)
        self.bytes["h2d"] += dst.numel() * dst.element_size()
        return dst

    def save(self, job: SaveJob) -> None:
        with self._on_stream():
            for c in range(0, len(job.slots), self.chunk):
                slots = job.slots[c:c + self.chunk]
                x = self._gather(job.gpu_blocks[c:c + self.chunk])
                self._to_host(self.bf16_pool, slots, x)
                if self.lo_pool is not None:
                    self._to_host(self.lo_pool, slots, pack_blocks(x, self.lo_bits))
            self._sending[job.req_id] = self._event()

    def load(self, job: LoadJob) -> None:
        with self._on_stream():
            for bits in sorted(set(job.bits)):
                sel = [i for i, b in enumerate(job.bits) if b == bits]
                for c in range(0, len(sel), self.chunk):
                    part = sel[c:c + self.chunk]
                    slots = [job.slots[i] for i in part]
                    if bits == 16:
                        data = self._to_device(self.bf16_pool, slots)
                    else:
                        if bits != self.lo_bits:
                            raise ValueError(f"no {bits}-bit copies stored (lo_bits={self.lo_bits})")
                        data = unpack_blocks(self._to_device(self.lo_pool, slots), bits, len(self.layers),
                                             self.heads, self.tokens, self.head_dim, self.dtype)
                    self._scatter([job.gpu_blocks[i] for i in part], data)
            self._recving[job.req_id] = self._event()

    def poll(self) -> tuple[set[str], set[str]]:
        """Request ids whose saves / loads have completed since the last poll."""
        def done(table):
            ids = {r for r, ev in table.items() if ev is None or ev.query()}
            for r in ids:
                del table[r]
            return ids
        return done(self._sending), done(self._recving)
