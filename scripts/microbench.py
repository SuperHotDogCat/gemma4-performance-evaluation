#!/usr/bin/env python3
"""Achievable (not datasheet) roofline on this GPU: BF16 GEMM TFLOPS and HBM bandwidth."""
from __future__ import annotations

import json
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parents[1] / "data" / "microbench.json"


def timeit(fn, iters=50, warm=10) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1e3


def main(out: Path = OUT) -> None:
    dev = torch.cuda.get_device_properties(0)
    res: dict = {"device": dev.name, "sms": dev.multi_processor_count,
                 "torch": torch.__version__, "cuda": torch.version.cuda}
    # HBM bandwidth: device-to-device copy of 4 GiB (read+write counted)
    n = 2 * 2**30 // 2
    a = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    t = timeit(lambda: b.copy_(a), iters=20)
    res["copy_bw_tbs"] = 2 * a.numel() * 2 / t / 1e12
    # read-only streaming (reduction) bandwidth
    t = timeit(lambda: a.sum(), iters=20)
    res["read_bw_tbs"] = a.numel() * 2 / t / 1e12
    del a, b
    # BF16 GEMM, square and decode-like skinny shapes (M = batch tokens)
    gemm = []
    for M, N, K in [(8192, 8192, 8192), (16384, 16384, 8192), (1, 21504, 5376), (8, 21504, 5376),
                    (64, 21504, 5376), (256, 21504, 5376), (1024, 21504, 5376), (4096, 21504, 5376)]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        t = timeit(lambda: x @ w)
        flops = 2 * M * N * K
        byts = 2 * (M * K + K * N + M * N)
        gemm.append({"M": M, "N": N, "K": K, "time_us": t * 1e6, "tflops": flops / t / 1e12,
                     "eff_bw_tbs": byts / t / 1e12, "arith_intensity": flops / byts})
        print(gemm[-1])
    res["gemm"] = gemm
    res["bf16_tflops_best"] = max(g["tflops"] for g in gemm)
    print(json.dumps({k: v for k, v in res.items() if k != "gemm"}, indent=1))
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    main(ap.parse_args().out)
