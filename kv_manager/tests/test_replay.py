import argparse
import asyncio
import json

from aiohttp import web

from bench import replay


async def _mock_server(seen):
    async def models(request):
        return web.json_response({"data": [{"id": "mock"}]})

    async def completions(request):
        body = await request.json()
        seen.append(body)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for i in range(body["max_tokens"]):
            await resp.write(f'data: {json.dumps({"choices": [{"text": "x"}]})}\n\n'.encode())
        usage = {"completion_tokens": body["max_tokens"], "prompt_tokens": len(body["prompt"])}
        await resp.write(f'data: {json.dumps({"choices": [], "usage": usage})}\n\n'.encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    app = web.Application()
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


def test_replay_against_mock_server(tmp_path):
    seen = []
    out = tmp_path / "run.jsonl"

    async def go():
        runner, port = await _mock_server(seen)
        args = argparse.Namespace(
            url=f"http://127.0.0.1:{port}", model=None, tokenizer="unsloth/Llama-3.1-8B-Instruct",
            agents=2, instances=["pallets__flask-5014"], max_turns=3, max_context=65536, gen_tokens=4,
            tool_delay_mean=0.01, tool_delay_min=0.0, stagger=0.0, tag=True, seed=0, out=str(out))
        try:
            await replay.main_async(args)
        finally:
            await runner.cleanup()

    asyncio.run(go())
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 6 and all(r["ttft"] is not None and r["gen_tokens"] == 4 for r in rows)
    for agent in (0, 1):
        lens = [r["prompt_tokens"] for r in sorted(rows, key=lambda r: r["turn"]) if r["agent"] == agent]
        assert lens == sorted(lens) and len(set(lens)) == 3
    assert {b["kv_transfer_params"]["tag"] for b in seen} == {f"a{a}-t{t}" for a in (0, 1) for t in range(3)}
    # Agents replaying the same recording must not share a prefix (different nonce in the system prompt).
    first = [b["prompt"] for b in seen if b["kv_transfer_params"]["tag"].endswith("-t0")]
    assert first[0][:20] != first[1][:20]
    assert "TTFT after tool return" in replay.summarize(out)
