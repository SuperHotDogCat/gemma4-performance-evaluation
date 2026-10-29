#!/usr/bin/env python3
"""
Online serving benchmark: launches `vllm serve`, streams requests through the
OpenAI-compatible API and records client-side TTFT / TPOT / ITL.

Why online instead of offline LLM.generate: RequestOutput.metrics timestamps in
vLLM v1 do not give a reliable TTFT (prefill got folded into "decode"), so we
measure what a user sees: time to first streamed content chunk.

Suites
  latency      concurrency=1, every workload sequentially (synthetic x3 repeats,
               Babilong 1k-32k, SWE-bench Lite genres). Spec-decode counters are
               scraped from /metrics around each request -> per-request acceptance.
  concurrency  SWE-bench prompts cycled, fixed 256 output tokens, sweep of
               concurrency levels -> throughput / TPOT / TTFT vs load.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_DIR = DATA / "serve"
LOG_DIR = ROOT / "logs"

SPEC_COUNTERS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


# --------------------------------------------------------------------------- server
def start_server(args, log_path: Path) -> subprocess.Popen:
    cmd = [
        "vllm", "serve", args.model,
        "--port", str(args.port),
        "--dtype", "bfloat16",
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_mem_util),
        "--max-num-seqs", str(args.max_num_seqs),
        # identical prompts are re-sent (repeats / concurrency sweep): a prefix
        # cache hit would make TTFT meaningless
        "--no-enable-prefix-caching",
        "--served-model-name", "m",
    ]
    if args.spec_method:
        sc: dict[str, Any] = {"method": args.spec_method,
                              "num_speculative_tokens": args.num_spec}
        if args.draft_model:
            sc["model"] = args.draft_model
        if args.spec_method == "ngram":
            sc.update(prompt_lookup_min=2, prompt_lookup_max=5)
        cmd += ["--speculative-config", json.dumps(sc)]
    cmd += args.extra_server_args
    print("$", " ".join(cmd), flush=True)
    f = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                            start_new_session=True, env=os.environ.copy())


def stop_server(p: subprocess.Popen) -> None:
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGINT)
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()


async def wait_ready(base: str, p: subprocess.Popen, timeout: float = 1800) -> None:
    t0 = time.time()
    async with aiohttp.ClientSession() as s:
        while time.time() - t0 < timeout:
            if p.poll() is not None:
                raise RuntimeError(f"vllm serve exited with {p.returncode}")
            try:
                async with s.get(f"{base}/health") as r:
                    if r.status == 200:
                        return
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(3)
    raise TimeoutError("server not ready")


async def scrape(s: aiohttp.ClientSession, base: str) -> dict[str, float]:
    async with s.get(f"{base}/metrics") as r:
        text = await r.text()
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
        if name in SPEC_COUNTERS:
            out[name] = out.get(name, 0.0) + val
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            pos = re.search(r'position="(\d+)"', labels)
            if pos:
                key = f"acc_pos_{pos.group(1)}"
                out[key] = out.get(key, 0.0) + val
    return out


def spec_delta(before: dict, after: dict) -> dict[str, Any] | None:
    if not after:
        return None
    d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
    drafts = d.get("vllm:spec_decode_num_drafts_total", 0.0)
    drafted = d.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    acc = d.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    pos = sorted((int(k.split("_")[-1]), v) for k, v in d.items() if k.startswith("acc_pos_"))
    return {
        "num_drafts": drafts,
        "num_draft_tokens": drafted,
        "num_accepted": acc,
        "acceptance_rate": acc / drafted if drafted else None,
        # tokens emitted per target forward = 1 bonus/corrected + accepted
        "mean_acceptance_length": 1 + acc / drafts if drafts else None,
        "acc_per_pos": [v / drafts if drafts else None for _, v in pos],
    }


# --------------------------------------------------------------------------- client
async def stream_one(s: aiohttp.ClientSession, base: str, prompt: str,
                     max_tokens: int, force_len: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "m",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": False},
        # Qwen3 hybrid-thinking: keep non-thinking mode; Gemma 4 thinks only if asked
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if force_len:
        # mask EOS until max_tokens -> fixed decode length while text stays natural
        body["min_tokens"] = max_tokens
    t0 = time.perf_counter()
    chunk_times: list[float] = []
    text_parts: list[str] = []
    usage = None
    async with s.post(f"{base}/v1/chat/completions", json=body) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:300]}")
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            j = json.loads(payload)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                c = ch.get("delta", {}).get("content")
                if c:
                    chunk_times.append(time.perf_counter())
                    text_parts.append(c)
    t_end = time.perf_counter()
    n_out = usage["completion_tokens"] if usage else len(chunk_times)
    n_in = usage["prompt_tokens"] if usage else None
    ttft = chunk_times[0] - t0 if chunk_times else float("nan")
    decode_s = (chunk_times[-1] - chunk_times[0]) if len(chunk_times) > 1 else float("nan")
    itl = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
    return {
        "prompt_tokens": n_in,
        "output_tokens": n_out,
        "ttft_s": ttft,
        "e2e_s": t_end - t0,
        "decode_s": decode_s,
        # time per output token after the first
        "tpot_s": decode_s / (n_out - 1) if n_out > 1 and decode_s == decode_s else None,
        "n_chunks": len(chunk_times),
        "itl_median_s": sorted(itl)[len(itl) // 2] if itl else None,
        "text": "".join(text_parts)[:400],
    }


def load_workloads() -> dict[str, list[dict]]:
    return json.loads((DATA / "workloads.json").read_text())


async def suite_latency(base: str, args) -> list[dict]:
    wl = load_workloads()
    items: list[tuple[dict, int, bool, int]] = []  # item, max_tokens, force_len, repeat
    for it in wl["synthetic"]:
        if it["max_tokens"] == 128:
            for rep in range(3):
                items.append((it, 128, True, rep))
    for it in wl["babilong"]:
        items.append((it, 128, True, 0))
    for it in wl["swe_bench_lite"]:
        items.append((it, 512, False, 0))  # natural generation: genre-dependent acceptance
    results = []
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        for i, (it, mt, force, rep) in enumerate(items):
            if it.get("approx_input_tokens", 0) + mt > args.max_model_len * 0.97:
                print(f"skip {it['id']} (exceeds max_model_len)")
                continue
            before = await scrape(s, base) if args.spec_method else {}
            try:
                r = await stream_one(s, base, it["prompt"], mt, force)
            except Exception as e:  # noqa: BLE001
                print("ERROR", it["id"], e, flush=True)
                continue
            after = await scrape(s, base) if args.spec_method else {}
            r.update(
                workload_id=it["id"], kind=it["kind"], repeat=rep, force_len=force,
                genre=it.get("genre"), length_config=it.get("length_config"),
                qa_split=it.get("qa_split"), target=it.get("target"),
                spec=spec_delta(before, after),
            )
            results.append(r)
            acc = r["spec"]["mean_acceptance_length"] if r["spec"] else None
            print(f"[{i+1}/{len(items)}] {it['id']:<40} in={r['prompt_tokens']:>6} "
                  f"out={r['output_tokens']:>4} TTFT={r['ttft_s']*1e3:8.1f}ms "
                  f"TPOT={(r['tpot_s'] or 0)*1e3:6.2f}ms"
                  + (f" AL={acc:.2f}" if acc else ""), flush=True)
    return results


async def suite_concurrency(base: str, args) -> list[dict]:
    prompts = [it["prompt"] for it in load_workloads()["swe_bench_lite"]]
    out_len = 256
    results = []
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout,
                                     connector=aiohttp.TCPConnector(limit=0)) as s:
        for c in args.concurrency:
            n_req = max(2 * c, 16)
            sem = asyncio.Semaphore(c)

            async def run(k: int):
                async with sem:
                    return await stream_one(s, base, prompts[k % len(prompts)], out_len, True)

            before = await scrape(s, base) if args.spec_method else {}
            t0 = time.perf_counter()
            reqs = await asyncio.gather(*(run(k) for k in range(n_req)), return_exceptions=True)
            wall = time.perf_counter() - t0
            after = await scrape(s, base) if args.spec_method else {}
            ok = [r for r in reqs if isinstance(r, dict)]
            errs = [str(r) for r in reqs if not isinstance(r, dict)]
            tot_out = sum(r["output_tokens"] for r in ok)
            tot_in = sum(r["prompt_tokens"] or 0 for r in ok)

            def med(key):
                v = sorted(x[key] for x in ok if x[key] is not None)
                return v[len(v) // 2] if v else None

            row = {
                "concurrency": c, "num_requests": n_req, "num_ok": len(ok),
                "errors": errs[:3], "wall_s": wall,
                "output_tok_s": tot_out / wall, "total_tok_s": (tot_out + tot_in) / wall,
                "req_s": len(ok) / wall,
                "median_ttft_s": med("ttft_s"), "median_tpot_s": med("tpot_s"),
                "mean_tpot_s": sum(x["tpot_s"] for x in ok if x["tpot_s"]) / max(len(ok), 1),
                "mean_prompt_tokens": tot_in / max(len(ok), 1),
                "spec": spec_delta(before, after),
            }
            results.append(row)
            print(f"c={c:>3} out_tok/s={row['output_tok_s']:8.1f} "
                  f"TTFT={row['median_ttft_s']*1e3:8.1f}ms TPOT={row['median_tpot_s']*1e3:6.2f}ms "
                  f"err={len(errs)}", flush=True)
    return results


async def warmup(base: str) -> None:
    wl = load_workloads()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
        for it in wl["synthetic"][:: 3][:4]:
            await stream_one(s, base, it["prompt"], 32, True)
        # a few concurrent ones to touch batched CUDA-graph shapes
        await asyncio.gather(*(stream_one(s, base, "Say hi.", 32, True) for _ in range(8)))


async def amain(args) -> None:
    out_dir, log_dir = Path(args.out_dir), Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    proc = start_server(args, log_dir / f"server_{args.tag}.log")
    try:
        t0 = time.time()
        await wait_ready(base, proc)
        startup = time.time() - t0
        print(f"server ready in {startup:.0f}s", flush=True)
        await warmup(base)
        payload: dict[str, Any] = {
            "tag": args.tag, "model": args.model, "draft_model": args.draft_model,
            "spec_method": args.spec_method, "num_speculative_tokens": args.num_spec,
            "max_model_len": args.max_model_len, "gpu_mem_util": args.gpu_mem_util,
            "startup_s": startup,
        }
        for suite in args.suites:
            print(f"=== {args.tag}: {suite} ===", flush=True)
            if suite == "latency":
                payload["latency"] = await suite_latency(base, args)
            elif suite == "concurrency":
                payload["concurrency"] = await suite_concurrency(base, args)
            out = out_dir / f"{args.tag}.json"
            out.write_text(json.dumps(payload, indent=1))
            print("wrote", out, flush=True)
    finally:
        stop_server(proc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--draft-model")
    ap.add_argument("--spec-method", choices=["mtp", "ngram", "eagle3", "draft_model"])
    ap.add_argument("--num-spec", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--suites", default="latency,concurrency",
                    type=lambda s: [x for x in s.split(",") if x])
    ap.add_argument("--concurrency", default="1,2,4,8,16,32,64",
                    type=lambda s: [int(x) for x in s.split(",")])
    # re-measurements go elsewhere: analyze.py reads every file in data/serve
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--log-dir", default=str(LOG_DIR))
    ap.add_argument("--extra-server-args", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()
    if not (DATA / "workloads.json").exists():
        sys.exit("data/workloads.json not found; run `python scripts/prepare_workloads.py` first")
    asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
