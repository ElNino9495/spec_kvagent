import pytest
import torch

from kvm.quant import block_bytes, dequantize, quantize, quantize_kv_block, roundtrip_blocks


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_pack_roundtrip_is_exact_on_grid(bits):
    levels = (1 << bits) - 1
    q = torch.randint(0, levels + 1, (3, 16, 128), dtype=torch.float32)
    q[..., 0, :], q[..., 1, :] = 0, levels  # pin min/max so the grid is exactly 0..levels
    x = q.to(torch.bfloat16)
    p = quantize(x, bits, reduce_dim=-2)
    assert torch.equal(dequantize(p, torch.float32), q)


@pytest.mark.parametrize("bits", [2, 4])
def test_error_shrinks_with_bits_and_packed_size_matches_accounting(bits):
    torch.manual_seed(0)
    heads, B, D = 8, 16, 128
    k = torch.randn(heads, B, D, dtype=torch.bfloat16) * 3
    v = torch.randn(heads, B, D, dtype=torch.bfloat16)
    pk, pv = quantize_kv_block(k, v, bits)
    assert pk.nbytes() + pv.nbytes() == block_bytes(bits, B, heads, D)
    err = (dequantize(pk) - k).float().abs().mean()
    err8 = (dequantize(quantize(k, 8, -2)) - k).float().abs().mean()
    assert err8 < err < k.float().abs().mean()


def test_roundtrip_blocks_keeps_partial_tail_and_matches_blockwise():
    torch.manual_seed(0)
    B = 16
    x = torch.randn(1, 8, 3 * B + 5, 128, dtype=torch.bfloat16)
    y = roundtrip_blocks(x, 2, B, "key")
    assert torch.equal(y[..., 3 * B:, :], x[..., 3 * B:, :])
    second = dequantize(quantize(x[..., B:2 * B, :], 2, -2), x.dtype)
    assert torch.equal(y[..., B:2 * B, :], second)


def test_block_bytes_ratios_for_llama_8b():
    bf16 = block_bytes(16, 16, 8, 128)
    assert bf16 == 16 * 8 * 128 * 4
    assert round(block_bytes(4, 16, 8, 128) / bf16, 3) == 0.320
    assert round(block_bytes(2, 16, 8, 128) / bf16, 3) == 0.195


@pytest.mark.parametrize("bits", [2, 4])
def test_pack_blocks_roundtrip_matches_per_tensor_quantization(bits):
    from kvm.quant import pack_blocks, unpack_blocks

    torch.manual_seed(0)
    n, L, H, B, D = 3, 2, 4, 16, 32
    x = torch.randn(n, L, H, B, 2 * D, dtype=torch.bfloat16)
    buf = pack_blocks(x, bits)
    assert buf.dtype == torch.uint8 and buf.shape == (n, L * block_bytes(bits, B, H, D))
    y = unpack_blocks(buf, bits, L, H, B, D)
    k_ref = dequantize(quantize(x[..., :D], bits, -2))
    v_ref = dequantize(quantize(x[..., D:], bits, -1))
    assert torch.equal(y, torch.cat([k_ref, v_ref], dim=-1))
