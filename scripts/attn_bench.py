#!/usr/bin/env python3
"""Isolated decode-attention benchmark with the shapes vLLM uses for Gemma 4 (see attn_probe.py).

One query token (batch 1), paged KV cache laid out like vLLM's (K and V interleaved per token),
pages in random order. Sweeps head dim, page size, KV length, num_splits and kernel
(FA4 = CuTe DSL FlashAttention used by vLLM 0.32 for Gemma 4, FA3, vLLM's Triton kernel).

  python scripts/attn_bench.py --out data/kernel/attn_bench.json             # full sweep, vLLM 0.32 as is
  python scripts/attn_bench.py --fix --out data/kernel/attn_bench_fix.json   # with the split-KV fix
  python scripts/attn_bench.py --only fa4:512:16:30015:0 --iters 3 --no-json  # one config (for ncu)
"""
from __future__ import annotations

import argparse
import json
import math

import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

ap = argparse.ArgumentParser()
ap.add_argument("--out")
ap.add_argument("--no-json", action="store_true")
ap.add_argument("--only", help="kernel:head_dim:page:seqlen:num_splits[:kv_heads]")
ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--fix", action="store_true", help="apply the split-KV fix (scripts/fa4fix)")
args = ap.parse_args()
if args.fix:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent / "fa4fix"))
    import fa4_splitkv_fix  # noqa: F401

dev = "cuda"
torch.manual_seed(0)
# Gemma 4 E2B: global layer 8 q heads / 1 kv head, head dim 512; local layer head dim 256.
NQ_PER_KV = 8


def make(hd, page, seqlen, kv_heads):
    n_pages = math.ceil(seqlen / page)
    total = n_pages + 64
    cache = torch.randn(total, page, 2, kv_heads, hd, device=dev, dtype=torch.bfloat16)
    k, v = cache[:, :, 0], cache[:, :, 1]
    table = torch.randperm(total, device=dev, dtype=torch.int32)[:n_pages].unsqueeze(0)
    q = torch.randn(1, kv_heads * NQ_PER_KV, hd, device=dev, dtype=torch.bfloat16)
    return q, k, v, table


def run_fa(ver, q, k, v, table, seqlen, splits, cu, su):
    return flash_attn_varlen_func(
        q, k, v, max_seqlen_q=1, cu_seqlens_q=cu, max_seqlen_k=seqlen, seqused_k=su,
        causal=True, block_table=table, softmax_scale=1.0 / math.sqrt(q.shape[-1]),
        fa_version=ver, num_splits=splits)


def run_triton(q, k, v, table, seqlen, splits, cu, su):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    out = torch.empty_like(q)
    unified_attention(
        q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=su, max_seqlen_k=seqlen,
        softmax_scale=1.0 / math.sqrt(q.shape[-1]), causal=True, window_size=(-1, -1),
        block_table=table, softcap=0, q_descale=None, k_descale=None, v_descale=None)
    return out


def reference(q, k, v, table, seqlen):
    hd = q.shape[-1]
    kk = k[table[0].long()].reshape(-1, k.shape[2], hd)[:seqlen].float()  # (S, Hkv, D)
    vv = v[table[0].long()].reshape(-1, v.shape[2], hd)[:seqlen].float()
    qq = q.float().view(k.shape[2], NQ_PER_KV, hd)
    s = torch.einsum("hgd,shd->hgs", qq, kk) / math.sqrt(hd)
    return torch.einsum("hgs,shd->hgd", s.softmax(-1), vv).reshape(q.shape)


def bench(kernel, hd, page, seqlen, splits, kv_heads=1):
    q, k, v, table = make(hd, page, seqlen, kv_heads)
    cu = torch.tensor([0, 1], device=dev, dtype=torch.int32)
    su = torch.tensor([seqlen], device=dev, dtype=torch.int32)
    fn = {"fa4": lambda: run_fa(4, q, k, v, table, seqlen, splits, cu, su),
          "fa3": lambda: run_fa(3, q, k, v, table, seqlen, splits, cu, su),
          "triton": lambda: run_triton(q, k, v, table, seqlen, splits, cu, su)}[kernel]  # 2D path only: not the split-KV path vLLM uses
    out = fn()
    err = (out.float() - reference(q, k, v, table, seqlen)).abs().max().item()
    torch.cuda.synchronize()
    # GPU time only, as under vLLM's CUDA graphs: replay a graph of R x (L2 flush + attention)
    # and subtract a graph of R x (L2 flush). Host-side dispatch of the CuTe DSL call
    # (~0.1 ms) would otherwise dominate short runs.
    l2_flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)  # > 60 MiB L2
    R = 10
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    graphs = []
    for with_attn in (True, False):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(R):
                l2_flush.zero_()
                if with_attn:
                    fn()
        graphs.append(g)

    def replay(g):
        ts = []
        for _ in range(args.iters):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1e3 / R)
        return sorted(ts)[len(ts) // 2]
    us = replay(graphs[0]) - replay(graphs[1])
    kv_bytes = 2 * seqlen * kv_heads * hd * 2
    rec = dict(kernel=kernel, fix=args.fix, head_dim=hd, page=page, seqlen=seqlen, num_splits=splits, kv_heads=kv_heads,
               us=us, kv_mb=kv_bytes / 1e6, gbs=kv_bytes / us / 1e3, max_abs_err=err)
    print(json.dumps(rec), flush=True)
    return rec


if args.only:
    kern, hd, page, sl, sp, *kvh = args.only.split(":")
    bench(kern, int(hd), int(page), int(sl), int(sp), int(kvh[0]) if kvh else 1)
    raise SystemExit

res = []
L = 30015
# Run once without and once with --fix; each run writes its own file.
# 1) the vLLM configuration (page 16, cp.async) vs. page size == tile_n (TMA path), split counts
for page in (16, 64):
    for sp in (0, 1, 8, 32, 128):
        res.append(bench("fa4", 512, page, L, sp))
# 2) head dim 256 (local layer geometry, no window) and FA3 on the same data
for page in (16, 32):
    res.append(bench("fa4", 256, page, L, 0))
res.append(bench("fa3", 256, 16, L, 0))
# 3) KV length sweep
for sl in (512, 1024, 2048, 4096, 8192, 16384):
    res.append(bench("fa4", 512, 16, sl, 0))
    res.append(bench("fa3", 256, 16, sl, 0))
# 4) 31B global layer: 4 kv heads
res.append(bench("fa4", 512, 16, 26000, 0, kv_heads=4))
if args.out and not args.no_json:
    json.dump(res, open(args.out, "w"), indent=1)
