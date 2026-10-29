#!/usr/bin/env python3
"""Build serving workloads from real HF datasets + a small synthetic grid.

Datasets:
  - RMT-team/babilong  — length configs 1k..32k, qa splits
  - SWE-bench/SWE-bench_Lite — test split, stratified by repo (genre)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parents[1] / "data" / "workloads.json"

FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "In a long-context evaluation, models must retrieve facts buried in noise. "
)

# Length configs available on RMT-team/babilong (skip 0k / huge 64k+)
BABILONG_LENGTHS = ["1k", "2k", "4k", "8k", "16k", "32k"]
# qa1 = single-fact / where-is style; vary qa for task diversity
BABILONG_QA = ["qa1", "qa2", "qa3"]
BABILONG_PER_CELL = 2  # samples per (length, qa)

# SWE-bench Lite: sample per top repos (= genre)
SWE_TOP_REPOS = [
    "django/django",
    "sympy/sympy",
    "matplotlib/matplotlib",
    "scikit-learn/scikit-learn",
    "pytest-dev/pytest",
    "sphinx-doc/sphinx",
    "astropy/astropy",
    "psf/requests",
]
SWE_PER_REPO = 3
SWE_MAX_PROMPT_CHARS = 120_000  # keep within typical max_model_len after tokenization


def swe_prompt(row: dict) -> str:
    """Serving-oriented prompt: problem + optional hints (no gold patch)."""
    hints = (row.get("hints_text") or "").strip()
    hints_block = f"\n\nHints from the issue discussion:\n{hints}\n" if hints else ""
    return (
        "You are a software engineering agent. Produce a minimal unified diff "
        "that fixes the bug described below. Do not explain; output the patch.\n\n"
        f"Repository: {row['repo']}\n"
        f"Instance: {row['instance_id']}\n"
        f"Difficulty: {row.get('difficulty', 'unknown')}\n\n"
        f"Problem statement:\n{row['problem_statement']}"
        f"{hints_block}"
    )


def babilong_prompt(row: dict) -> str:
    return (
        "Read the document carefully and answer the question with a short phrase.\n\n"
        f"Document:\n{row['input']}\n\n"
        f"Question: {row['question']}\n"
        "Answer:"
    )


def build_synthetic() -> list[dict]:
    items = []
    for inp in [256, 512, 1024, 2048, 4096, 8192, 16384]:
        for out in [128, 256, 512]:
            items.append(
                {
                    "kind": "synthetic",
                    "id": f"syn_in{inp}_out{out}",
                    "approx_input_tokens": inp,
                    "max_tokens": out,
                    "prompt": (
                        "Write a detailed technical explanation of GPU "
                        "memory-bandwidth-bound LLM decoding, including KV cache "
                        "and weight traffic. " + FILLER * (inp // 20)
                    ),
                }
            )
    return items


def build_babilong() -> list[dict]:
    items = []
    for length in BABILONG_LENGTHS:
        # config name is length; splits are qa1..qa10
        for qa in BABILONG_QA:
            ds = load_dataset("RMT-team/babilong", length, split=qa)
            n = min(BABILONG_PER_CELL, len(ds))
            for i in range(n):
                row = ds[i]
                # rough token estimate from chars
                approx = max(1, len(row["input"]) // 4)
                items.append(
                    {
                        "kind": "babilong",
                        "id": f"bab_{length}_{qa}_{i}",
                        "length_config": length,
                        "qa_split": qa,
                        "approx_input_tokens": approx,
                        "max_tokens": 32,
                        "target": row["target"],
                        "prompt": babilong_prompt(row),
                        "dataset": "RMT-team/babilong",
                    }
                )
    return items


def build_swe_lite() -> list[dict]:
    ds = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
    by_repo: dict[str, list] = defaultdict(list)
    for row in ds:
        by_repo[row["repo"]].append(row)

    items = []
    for repo in SWE_TOP_REPOS:
        rows = by_repo.get(repo, [])
        if not rows:
            continue
        for i, row in enumerate(rows[:SWE_PER_REPO]):
            prompt = swe_prompt(row)
            if len(prompt) > SWE_MAX_PROMPT_CHARS:
                prompt = prompt[:SWE_MAX_PROMPT_CHARS] + "\n\n[truncated]\n"
            genre = repo.split("/")[0]  # django, sympy, ...
            items.append(
                {
                    "kind": "swe_bench_lite",
                    "id": f"swe_{row['instance_id']}",
                    "genre": genre,
                    "repo": repo,
                    "instance_id": row["instance_id"],
                    "difficulty": row.get("difficulty"),
                    "approx_input_tokens": max(1, len(prompt) // 4),
                    "max_tokens": 512,
                    "prompt": prompt,
                    "dataset": "SWE-bench/SWE-bench_Lite",
                }
            )
    return items


def main() -> None:
    print("Loading RMT-team/babilong ...")
    babilong = build_babilong()
    print(f"  {len(babilong)} samples")

    print("Loading SWE-bench/SWE-bench_Lite ...")
    swe = build_swe_lite()
    print(f"  {len(swe)} samples across repos:", sorted({x['repo'] for x in swe}))

    synthetic = build_synthetic()
    payload = {
        "synthetic": synthetic,
        "babilong": babilong,
        "swe_bench_lite": swe,
        "meta": {
            "babilong_dataset": "RMT-team/babilong",
            "babilong_lengths": BABILONG_LENGTHS,
            "babilong_qa": BABILONG_QA,
            "swe_dataset": "SWE-bench/SWE-bench_Lite",
            "swe_split": "test",
            "swe_repos": SWE_TOP_REPOS,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}")
    print("counts:", {k: len(v) if isinstance(v, list) else v for k, v in payload.items()})


if __name__ == "__main__":
    main()
