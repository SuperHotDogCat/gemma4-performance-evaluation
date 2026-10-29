#!/usr/bin/env python3
"""Decode-phase workload for Nsight Systems (run under `nsys --capture-range=cudaProfilerApi`).

Engine runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so the profiler range
brackets exactly the generate() call. Prompts are short so the captured window is
>95% decode steps.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--draft-model")
ap.add_argument("--num-spec", type=int, default=4)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--out-tokens", type=int, default=256)
ap.add_argument("--max-model-len", type=int, default=8192)
ap.add_argument("--gpu-mem-util", type=float, default=0.90)
ap.add_argument("--enforce-eager", action="store_true")
ap.add_argument("--context", type=int, default=0, help="pad prompt to ~N tokens (long-context decode)")
args = ap.parse_args()

kw = dict(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
          gpu_memory_utilization=args.gpu_mem_util, enable_prefix_caching=False,
          enforce_eager=args.enforce_eager)
if args.draft_model:
    kw["speculative_config"] = {"method": "mtp", "model": args.draft_model,
                                "num_speculative_tokens": args.num_spec}
llm = LLM(**kw)
text = "Explain, step by step, how a GPU executes a transformer decoder layer."
if args.context:
    filler = "The quick brown fox jumps over the lazy dog. "  # ~10 tokens
    text = filler * (args.context // 10) + text
prompt = [{"role": "user", "content": text}]
sp = SamplingParams(temperature=0.0, max_tokens=args.out_tokens,
                    min_tokens=args.out_tokens, ignore_eos=True)
batch = [prompt] * args.batch
llm.chat(batch, sp, use_tqdm=False)  # warm-up (same shapes)

torch.cuda.synchronize()
torch.cuda.profiler.start()
t0 = time.perf_counter()
outs = llm.chat(batch, sp, use_tqdm=False)
torch.cuda.synchronize()
wall = time.perf_counter() - t0
torch.cuda.profiler.stop()
n = sum(len(o.outputs[0].token_ids) for o in outs)
print("PROFILE_RESULT", json.dumps({"wall_s": wall, "tokens": n, "batch": args.batch,
                                     "tok_s": n / wall}))
