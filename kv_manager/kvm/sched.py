"""Scheduler-side bookkeeping for AgentKVConnector (pure Python, no vLLM imports).

CPU slots are keyed by vLLM's prefix-cache block hashes, so the store is content addressed:
an agent's next turn shares the hash chain of its previous turns and hits automatically.
A slot holds one full block (all layers) in bf16 and, unless mode is "bf16", a packed
low-bit copy. Saved slots are immutable, and a block that was ever reloaded at low precision
is never saved again (vLLM keeps the lossy GPU copy under the original hash), so the CPU
store only ever holds originals.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Hashable

MODES = {"bf16": None, "uniform_int4": 4, "uniform_int2": 2}


@dataclass
class LoadJob:
    req_id: str
    gpu_blocks: list[int]
    slots: list[int]
    bits: list[int]  # 16 = bf16 copy, otherwise the packed low-bit copy


@dataclass
class SaveJob:
    req_id: str
    gpu_blocks: list[int]
    slots: list[int]


@dataclass
class _Entry:
    slot: int
    ready: bool = False
    pins: int = 0


class BlockIndex:
    """block hash -> CPU slot, with LRU eviction of ready, unpinned blocks."""

    def __init__(self, n_slots: int):
        self.free = list(range(n_slots))
        self.entries: OrderedDict[Hashable, _Entry] = OrderedDict()

    def ready_slot(self, h: Hashable) -> int | None:
        e = self.entries.get(h)
        return e.slot if e is not None and e.ready else None

    def __contains__(self, h: Hashable) -> bool:
        return h in self.entries

    def allocate(self, hashes: list[Hashable]) -> list[int] | None:
        """Reserve one slot per hash (entries start not-ready). None if it cannot fit."""
        need = len(hashes) - len(self.free)
        if need > 0:
            victims = [h for h, e in self.entries.items() if e.ready and e.pins == 0][:need]
            if len(victims) < need:
                return None
            for h in victims:
                self.free.append(self.entries.pop(h).slot)
            self.free.sort()
        slots, self.free = self.free[:len(hashes)], self.free[len(hashes):]
        for h, s in zip(hashes, slots):
            self.entries[h] = _Entry(s)
        return slots

    def mark_ready(self, hashes: list[Hashable]) -> None:
        for h in hashes:
            if h in self.entries:
                self.entries[h].ready = True

    def pin(self, hashes: list[Hashable], delta: int) -> None:
        for h in hashes:
            e = self.entries.get(h)
            if e is not None:
                e.pins += delta
                self.entries.move_to_end(h)


def plan_bits(first: int, n: int, total: int, lo_bits: int | None, recent_blocks: int) -> list[int]:
    """Precision for history blocks first..first+n-1 out of `total` matched blocks.

    Block 0 (attention sinks) and the last `recent_blocks` blocks stay bf16; with
    lo_bits=None everything is bf16.
    """
    if lo_bits is None:
        return [16] * n
    return [16 if j == 0 or j >= total - recent_blocks else lo_bits for j in range(first, first + n)]


@dataclass
class SchedulerState:
    block_size: int
    n_slots: int
    mode: str
    recent_blocks: int
    bytes_per_block: dict[int, int]  # bits -> bytes for one block, all layers
    index: BlockIndex = field(init=False)
    loads: list[LoadJob] = field(default_factory=list)
    saves: list[SaveJob] = field(default_factory=list)
    _local: dict[str, int] = field(default_factory=dict)
    _loading: dict[str, list[Hashable]] = field(default_factory=dict)
    _saving: dict[str, list[Hashable]] = field(default_factory=dict)
    _lossy: set[Hashable] = field(default_factory=set)
    stats: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}, got {self.mode!r}")
        self.lo_bits = MODES[self.mode]
        self.index = BlockIndex(self.n_slots)

    def _matched_blocks(self, hashes: list[Hashable], start: int) -> int:
        end = start
        while end < len(hashes) and self.index.ready_slot(hashes[end]) is not None:
            end += 1
        return end

    def num_matched_tokens(self, req_id: str, hashes: list[Hashable], num_computed: int) -> int:
        """Tokens loadable from CPU beyond the block-aligned local prefix-cache hit."""
        start = num_computed // self.block_size
        self._local[req_id] = start
        return (self._matched_blocks(hashes, start) - start) * self.block_size

    def on_alloc(self, req_id: str, hashes: list[Hashable], gpu_blocks: list[int],
                 num_external_tokens: int) -> None:
        if num_external_tokens == 0:
            return
        start = self._local[req_id]
        n = num_external_tokens // self.block_size
        total = self._matched_blocks(hashes, start)
        keys = hashes[start:start + n]
        bits = plan_bits(start, n, total, self.lo_bits, self.recent_blocks)
        self.index.pin(keys, +1)
        self._loading[req_id] = keys
        # vLLM caches these GPU blocks under their exact hash; never save them back as originals.
        self._lossy.update(h for h, b in zip(keys, bits) if b != 16)
        self.loads.append(LoadJob(req_id, gpu_blocks[start:start + n],
                                  [self.index.ready_slot(h) for h in keys], bits))
        n_hi = bits.count(16)
        self.stats[req_id] = {"loaded_blocks": n, "bf16_blocks": n_hi, "lo_blocks": n - n_hi,
                              "load_bytes": sum(self.bytes_per_block[b] for b in bits)}

    def on_finished(self, req_id: str, hashes: list[Hashable], gpu_blocks: list[int]) -> tuple[bool, dict]:
        """Queue a save of every full block not stored yet.

        Returns (delay freeing the blocks until the save is done, per-request stats).
        """
        self._local.pop(req_id, None)
        stats = self.stats.pop(req_id, {"loaded_blocks": 0, "load_bytes": 0})
        n = min(len(hashes), len(gpu_blocks))
        new = [j for j in range(n) if hashes[j] not in self.index and hashes[j] not in self._lossy]
        stats["saved_blocks"] = 0
        if not new or req_id in self._loading:
            # Aborted while its load is in flight: vLLM already delays the free until
            # finished_recving, and a second (sending) completion would be one too many.
            return False, stats
        slots = self.index.allocate([hashes[j] for j in new])
        if slots is None:
            return False, stats
        self._saving[req_id] = [hashes[j] for j in new]
        self.saves.append(SaveJob(req_id, [gpu_blocks[j] for j in new], slots))
        stats["saved_blocks"] = len(new)
        return True, stats

    def on_worker_output(self, finished_sending, finished_recving) -> None:
        for req_id in finished_sending or ():
            self.index.mark_ready(self._saving.pop(req_id, []))
        for req_id in finished_recving or ():
            self.index.pin(self._loading.pop(req_id, []), -1)

    def take_jobs(self) -> tuple[list[LoadJob], list[SaveJob]]:
        loads, saves = self.loads, self.saves
        self.loads, self.saves = [], []
        return loads, saves
