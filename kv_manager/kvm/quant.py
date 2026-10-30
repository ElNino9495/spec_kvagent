"""Block-wise KIVI-style KV-cache quantization in plain PyTorch (no custom kernels).

A KV block is `block_tokens` consecutive tokens of one layer. Tensors use the
(..., tokens, head_dim) layout shared by HF caches and vLLM blocks.

- Keys:   per-channel min/max over the block's tokens (keys have outlier channels).
- Values: per-token min/max over head_dim.
- Scale and zero-point (the min) are stored as fp16; codes are packed 8/bits per byte.

The offline gate experiment and the vLLM connector use the same functions, so the
gate measures exactly the format the system ships.
"""

from dataclasses import dataclass

import torch

KEY_REDUCE_DIM = -2
VALUE_REDUCE_DIM = -1


@dataclass
class Packed:
    data: torch.Tensor   # uint8, last dim packed 8 // bits codes per byte
    scale: torch.Tensor  # fp16
    zero: torch.Tensor   # fp16
    bits: int
    shape: torch.Size

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.data, self.scale, self.zero))


def _pack(q: torch.Tensor, bits: int) -> torch.Tensor:
    per = 8 // bits
    q = q.reshape(*q.shape[:-1], q.shape[-1] // per, per)
    out = q[..., 0].clone()
    for i in range(1, per):
        out |= q[..., i] << (bits * i)
    return out


def _unpack(p: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    per = 8 // bits
    mask = (1 << bits) - 1
    parts = [(p >> (bits * i)) & mask for i in range(per)]
    return torch.stack(parts, dim=-1).reshape(*p.shape[:-1], n)


def quantize(x: torch.Tensor, bits: int, reduce_dim: int) -> Packed:
    if bits not in (2, 4, 8):
        raise ValueError(f"bits must be 2, 4 or 8, got {bits}")
    if x.shape[-1] % (8 // bits):
        raise ValueError(f"last dim {x.shape[-1]} not divisible by {8 // bits}")
    xf = x.float()
    mn = xf.amin(dim=reduce_dim, keepdim=True)
    mx = xf.amax(dim=reduce_dim, keepdim=True)
    levels = (1 << bits) - 1
    scale = ((mx - mn) / levels).clamp(min=1e-5).half()
    zero = mn.half()
    q = ((xf - zero.float()) / scale.float()).round().clamp(0, levels).to(torch.uint8)
    return Packed(_pack(q, bits), scale, zero, bits, x.shape)


def dequantize(p: Packed, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    q = _unpack(p.data, p.bits, p.shape[-1]).float()
    return (q * p.scale.float() + p.zero.float()).to(dtype)


def quantize_kv_block(k: torch.Tensor, v: torch.Tensor, bits: int) -> tuple[Packed, Packed]:
    return quantize(k, bits, KEY_REDUCE_DIM), quantize(v, bits, VALUE_REDUCE_DIM)


def roundtrip_blocks(x: torch.Tensor, bits: int, block_tokens: int, kind: str) -> torch.Tensor:
    """Quantize -> dequantize every full block of x (..., tokens, head_dim) along the token axis.

    The trailing partial block is returned unchanged: it is never offloaded (vLLM only
    caches full blocks) and is recomputed in full precision on the next turn.
    """
    reduce_dim = {"key": KEY_REDUCE_DIM, "value": VALUE_REDUCE_DIM}[kind]
    n_tok, d = x.shape[-2], x.shape[-1]
    full = (n_tok // block_tokens) * block_tokens
    out = x.clone()
    if full:
        xb = x[..., :full, :].reshape(*x.shape[:-2], full // block_tokens, block_tokens, d)
        deq = dequantize(quantize(xb, bits, reduce_dim), x.dtype)
        out[..., :full, :] = deq.reshape(*x.shape[:-2], full, d)
    return out


def block_bytes(bits: int, block_tokens: int, kv_heads: int, head_dim: int) -> int:
    """Bytes for one block of one layer (K + V), including fp16 scales and zero-points.

    bits=16 means unquantized bf16/fp16.
    """
    elems = kv_heads * block_tokens * head_dim
    if bits == 16:
        return 2 * elems * 2
    codes = elems * bits // 8
    key_meta = kv_heads * head_dim * 2 * 2        # per-channel scale + zero
    value_meta = kv_heads * block_tokens * 2 * 2  # per-token scale + zero
    return 2 * codes + key_meta + value_meta


def _packed_parts(n: int, layers: int, heads: int, tokens: int, head_dim: int, bits: int):
    codes = (n, layers, heads, tokens, head_dim * bits // 8)
    return [("k_data", codes, torch.uint8), ("k_scale", (n, layers, heads, 1, head_dim), torch.float16),
            ("k_zero", (n, layers, heads, 1, head_dim), torch.float16),
            ("v_data", codes, torch.uint8), ("v_scale", (n, layers, heads, tokens, 1), torch.float16),
            ("v_zero", (n, layers, heads, tokens, 1), torch.float16)]


def pack_blocks(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize whole KV blocks into one flat byte row per block.

    x: (n, layers, heads, tokens, 2 * head_dim), keys in the first head_dim channels and
    values in the second (vLLM's FlashAttention block layout). Returns uint8 (n, nbytes)
    with nbytes == layers * block_bytes(bits, tokens, heads, head_dim).
    """
    d = x.shape[-1] // 2
    pk, pv = quantize_kv_block(x[..., :d], x[..., d:], bits)
    parts = [pk.data, pk.scale, pk.zero, pv.data, pv.scale, pv.zero]
    return torch.cat([p.reshape(x.shape[0], -1).view(torch.uint8) for p in parts], dim=1)


def unpack_blocks(buf: torch.Tensor, bits: int, layers: int, heads: int, tokens: int,
                  head_dim: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Inverse of pack_blocks: uint8 (n, nbytes) -> (n, layers, heads, tokens, 2 * head_dim)."""
    n, off, t = buf.shape[0], 0, {}
    for name, shape, dt in _packed_parts(n, layers, heads, tokens, head_dim, bits):
        size = 1
        for s in shape[1:]:
            size *= s
        size *= torch.tensor([], dtype=dt).element_size()
        t[name] = buf[:, off:off + size].contiguous().view(dt).reshape(shape)
        off += size
    if off != buf.shape[1]:
        raise ValueError(f"packed row has {buf.shape[1]} bytes, expected {off}")
    shape = torch.Size((n, layers, heads, tokens, head_dim))
    k = dequantize(Packed(t["k_data"], t["k_scale"], t["k_zero"], bits, shape), dtype)
    v = dequantize(Packed(t["v_data"], t["v_scale"], t["v_zero"], bits, shape), dtype)
    return torch.cat([k, v], dim=-1)
