#!/usr/bin/env python3
"""Summarize an nsys SQLite export: GPU busy fraction and kernel-time breakdown by category."""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

CATS = [
    ("moe", r"moe|expert|topk_softmax|grouped|align_block"),
    ("attention", r"flash|fmha|attn|attention|paged|cute.*fwd|fwd_sm90|decode_kernel|split_kv"),
    ("gemm", r"gemm|nvjet|cutlass|sm90_xmma|cublas|matmul|splitk|wgmma"),
    ("sampling/logits", r"softmax|argmax|sample|topk|top_k|sort|softcap|logit|gather|scatter"),
    ("norm/elementwise", r"norm|rms|elementwise|vectorized|triton_|add|mul|gelu|silu|rotary|rope|cat|copy|fill|reduce"),
]


def categorize(name: str) -> str:
    low = name.lower()
    for cat, pat in CATS:
        if re.search(pat, low):
            return cat
    return "other"


def analyze(db_path: Path) -> dict:
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT k.start, k.end, s.value, k.gridX * k.gridY * k.gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.demangledName = s.id ORDER BY k.start").fetchall()
    if not rows:
        raise SystemExit("no kernels")
    t0, t1 = rows[0][0], max(r[1] for r in rows)
    busy, cur_s, cur_e = 0, None, None
    by_cat: dict[str, float] = {}
    by_kernel: dict[str, list] = {}
    for s, e, name, grid in rows:
        d = e - s
        c = categorize(name)
        by_cat[c] = by_cat.get(c, 0) + d
        k = by_kernel.setdefault(name[:120], [0, 0, c, 0])
        k[0] += d
        k[1] += 1
        k[3] += grid
        if cur_s is None or s > cur_e:
            if cur_s is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    span = t1 - t0
    total_k = sum(by_cat.values())
    top = sorted(by_kernel.items(), key=lambda x: -x[1][0])[:15]
    return {
        "span_ms": span / 1e6,
        "gpu_busy_ms": busy / 1e6,
        "gpu_busy_frac": busy / span,
        "num_kernels": len(rows),
        "kernel_time_by_cat_frac": {k: v / total_k for k, v in sorted(by_cat.items(), key=lambda x: -x[1])},
        "kernel_time_by_cat_ms": {k: v / 1e6 for k, v in by_cat.items()},
        "top_kernels": [{"name": n, "ms": v[0] / 1e6, "count": v[1], "cat": v[2],
                         "avg_us": v[0] / v[1] / 1e3, "avg_ctas": v[3] / v[1]} for n, v in top],
    }


if __name__ == "__main__":
    res = analyze(Path(sys.argv[1]))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "top_kernels"}, indent=1))
    for t in res["top_kernels"]:
        print(f"{t['ms']:9.2f} ms {t['count']:6d}x {t['avg_us']:8.1f}us ctas={t['avg_ctas']:7.0f} [{t['cat']}] {t['name'][:90]}")
