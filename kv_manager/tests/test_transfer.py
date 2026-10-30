import pytest
import torch

from kvm.quant import pack_blocks, unpack_blocks
from kvm.sched import LoadJob, SaveJob
from kvm.transfer import KVTransfer, _runs

L, NB, H, B, D = 3, 20, 2, 16, 32


def make_layers():
    torch.manual_seed(0)
    full = torch.randn(L, NB, B, H, 2 * D, dtype=torch.bfloat16)
    # vLLM hands out a permuted (non-contiguous) logical view: (num_blocks, H, B, 2D)
    return [full[i].permute(0, 2, 1, 3) for i in range(L)]


def test_runs_groups_consecutive_slots():
    assert list(_runs([4, 5, 6, 9, 2, 3])) == [(0, 4, 3), (3, 9, 1), (4, 2, 2)]


@pytest.mark.parametrize("lo_bits", [2, 4])
def test_save_then_mixed_precision_load_on_cpu(lo_bits):
    layers = make_layers()
    orig = [t.clone() for t in layers]
    xfer = KVTransfer(layers, n_slots=8, lo_bits=lo_bits, chunk_blocks=2)
    src = [3, 5, 6, 7]
    xfer.save(SaveJob("s", src, [4, 5, 6, 0]))
    assert xfer.poll() == ({"s"}, set())

    dst = [10, 11, 12, 13]
    for t in layers:
        t[dst] = 0
    xfer.load(LoadJob("l", dst, [4, 5, 6, 0], [16, lo_bits, lo_bits, 16]))
    assert xfer.poll() == (set(), {"l"})

    stacked = lambda ts, ids: torch.stack([t[ids] for t in ts], dim=1)
    assert torch.equal(stacked(layers, [10, 13]), stacked(orig, [3, 7]))
    expect = unpack_blocks(pack_blocks(stacked(orig, [5, 6]), lo_bits), lo_bits, L, H, B, D)
    assert torch.equal(stacked(layers, [11, 12]), expect)
    untouched = [i for i in range(NB) if i not in dst]
    assert all(torch.equal(t[untouched], o[untouched]) for t, o in zip(layers, orig))


def test_loading_bits_without_stored_copy_fails():
    xfer = KVTransfer(make_layers(), n_slots=2, lo_bits=None)
    xfer.save(SaveJob("s", [1], [0]))
    with pytest.raises(ValueError):
        xfer.load(LoadJob("l", [2], [0], [2]))
