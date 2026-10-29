#!/usr/bin/env python3
"""Decode-shaped GEMM (M = 1) time vs. weight size inside a CUDA graph, plus the per-kernel
floor of an empty kernel. Explains why small models cannot reach HBM bandwidth in decode.

Weights are cycled through a ring larger than L2 so every call reads from HBM.
  python scripts/gemv_bench.py --out data/kernel/gemv_bench.json
"""
from __future__ import annotations

import argparse
import json

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
args = ap.parse_args()
dev = "cuda"
R = 200  # calls per graph


def graph_time_us(fn, iters=20) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(R):
            fn(i)
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / R)
    return sorted(ts)[len(ts) // 2]


res = {}
x1 = torch.zeros(1, device=dev)
res["empty_kernel_us"] = graph_time_us(lambda i: x1.add_(1))
print("empty kernel", res["empty_kernel_us"])

rows = []
for K in (1536, 5376):
    for N in (256, 512, 1024, 1536, 2048, 2560, 4096, 6144, 8192, 12288, 16384, 21504, 43008, 86016):
        nbytes = N * K * 2
        ring = max(2, min(64, -(-256 * 2**20 // nbytes)))  # >= 256 MiB of distinct weights
        ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) for _ in range(ring)]
        x = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
        us = graph_time_us(lambda i: torch.nn.functional.linear(x, ws[i % ring]))
        rows.append(dict(N=N, K=K, mb=nbytes / 1e6, us=us, gbs=nbytes / us / 1e3))
        print(rows[-1], flush=True)
        del ws
        torch.cuda.empty_cache()
res["gemv"] = rows
json.dump(res, open(args.out, "w"), indent=1)
