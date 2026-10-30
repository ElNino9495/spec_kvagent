from kvm.sched import BlockIndex, SchedulerState, plan_bits

B = 16


def make_state(n_slots=8, mode="uniform_int2", recent=1):
    return SchedulerState(B, n_slots, mode, recent, {16: 100, 2: 20, 4: 30})


def finish_and_save(state, req, hashes, gpu):
    delay, stats = state.on_finished(req, hashes, gpu)
    loads, saves = state.take_jobs()
    state.on_worker_output({req} if saves else None, None)
    return delay, stats, saves


def test_plan_bits_keeps_sink_and_recent_window_bf16():
    assert plan_bits(0, 6, 6, 2, 2) == [16, 2, 2, 2, 16, 16]
    assert plan_bits(2, 3, 6, 4, 1) == [4, 4, 4]
    assert plan_bits(3, 3, 6, 4, 1) == [4, 4, 16]
    assert plan_bits(0, 3, 3, None, 1) == [16, 16, 16]


def test_block_index_evicts_lru_ready_unpinned_only():
    idx = BlockIndex(3)
    assert idx.allocate(["a", "b", "c"]) == [0, 1, 2]
    assert idx.allocate(["d"]) is None  # nothing ready yet
    idx.mark_ready(["a", "b", "c"])
    idx.pin(["a"], +1)  # a is being loaded, also becomes most recently used
    assert idx.allocate(["d", "e"]) == [1, 2]  # evicts b, c (a is pinned)
    assert "a" in idx and "b" not in idx and "c" not in idx
    assert idx.allocate(["f"]) is None  # d, e not ready, a pinned


def test_turn_cycle_saves_then_loads_with_precision_plan():
    s = make_state()
    hashes = [f"h{i}" for i in range(4)]
    assert s.num_matched_tokens("t1", hashes, 0) == 0
    delay, stats, saves = finish_and_save(s, "t1", hashes, [10, 11, 12, 13])
    assert delay and stats["saved_blocks"] == 4 and saves[0].slots == [0, 1, 2, 3]

    # Next turn: same 4 blocks + 2 new ones; blocks 0-1 still in the GPU prefix cache.
    nxt = hashes + ["h4", "h5"]
    assert s.num_matched_tokens("t2", nxt, 2 * B) == 2 * B
    s.on_alloc("t2", nxt, [20, 21, 22, 23, 24, 25], 2 * B)
    loads, _ = s.take_jobs()
    (job,) = loads
    assert job.gpu_blocks == [22, 23] and job.slots == [2, 3]
    assert job.bits == [2, 16]  # block 3 is in the recent window
    assert s.index.entries["h2"].pins == 1
    s.on_worker_output(None, {"t2"})
    assert s.index.entries["h2"].pins == 0

    # Finishing t2 saves only the 2 new blocks and reports the load stats.
    delay, stats, saves = finish_and_save(s, "t2", nxt, [20, 21, 22, 23, 24, 25])
    assert delay and saves[0].gpu_blocks == [24, 25]
    assert stats["loaded_blocks"] == 2 and stats["load_bytes"] == 120 and stats["saved_blocks"] == 2


def test_unready_blocks_are_not_matched_and_nothing_new_means_no_delay():
    s = make_state()
    hashes = ["a", "b"]
    s.on_finished("t1", hashes, [1, 2])
    s.take_jobs()
    assert s.num_matched_tokens("t2", hashes, 0) == 0  # save still in flight
    s.on_worker_output({"t1"}, None)
    assert s.num_matched_tokens("t2", hashes, 0) == 2 * B
    delay, stats = s.on_finished("t3", hashes, [1, 2])
    assert not delay and stats["saved_blocks"] == 0


def test_bf16_mode_loads_everything_in_bf16():
    s = make_state(mode="bf16")
    hashes = [f"h{i}" for i in range(5)]
    finish_and_save(s, "t1", hashes, list(range(5)))
    s.num_matched_tokens("t2", hashes, 0)
    s.on_alloc("t2", hashes, list(range(10, 15)), 5 * B)
    (job,), _ = s.take_jobs()
    assert job.bits == [16] * 5


def _load(s, req, hashes, local_blocks, gpu):
    n = s.num_matched_tokens(req, hashes, local_blocks * B)
    s.on_alloc(req, hashes, gpu, n)
    loads, _ = s.take_jobs()
    return loads


def test_abort_during_load_does_not_also_save():
    s = make_state()
    finish_and_save(s, "t1", ["h0", "h1", "h2", "h3"], [0, 1, 2, 3])
    s.index.entries.pop("h0")  # h0 evicted from CPU but still a local GPU hit
    _load(s, "t2", ["h0", "h1", "h2", "h3"], 1, [9, 10, 11, 12])
    delay, _ = s.on_finished("t2", ["h0", "h1", "h2", "h3"], [9, 10, 11, 12])
    assert not delay and s.take_jobs() == ([], [])
    s.on_worker_output(None, {"t2"})
    assert all(e.pins == 0 for e in s.index.entries.values())


def test_lossy_reloads_are_never_saved_back_as_originals():
    s = make_state(n_slots=4, recent=1)
    hashes = ["h0", "h1", "h2", "h3"]
    finish_and_save(s, "t1", hashes, [0, 1, 2, 3])
    (job,) = _load(s, "t2", hashes, 0, [10, 11, 12, 13])
    assert job.bits == [16, 2, 2, 16]
    s.on_worker_output(None, {"t2"})
    s.on_finished("t2", hashes, [10, 11, 12, 13])
    for h in hashes:  # CPU copies get evicted; GPU still holds the lossy h1/h2
        s.index.entries.pop(h)
        s.index.free.append(len(s.index.free))
    delay, stats, saves = finish_and_save(s, "t3", hashes, [10, 11, 12, 13])
    assert saves[0].gpu_blocks == [10, 13] and stats["saved_blocks"] == 2
