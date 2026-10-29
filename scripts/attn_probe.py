#!/usr/bin/env python3
"""Record the arguments vLLM passes to the FA4 attention kernel during long-context decode.

Wraps the FA4 dense-attention dispatcher in vllm/v1/attention/backends/flash_attn.py and
prints one line per distinct call signature (head dim, num_splits, page size, ...), so the
kernel can be reproduced in isolation by attn_bench.py.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402
import vllm.v1.attention.backends.flash_attn as fa  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--context", type=int, default=30000)
ap.add_argument("--max-model-len", type=int, default=34000)
ap.add_argument("--gpu-mem-util", type=float, default=0.90)
ap.add_argument("--out", required=True)
args = ap.parse_args()

seen: dict[str, dict] = {}
kernel_cls = type(fa._FA4_DENSE_ATTENTION_KERNEL)
orig_call = kernel_cls.__call__


def desc(x):
    if isinstance(x, torch.Tensor):
        return {"shape": list(x.shape), "stride": list(x.stride()), "dtype": str(x.dtype)}
    return x if isinstance(x, (int, float, bool, str, type(None), list, tuple)) else type(x).__name__


def logged_call(self, *, q, k, v, **kw):
    rec = {"q": desc(q), "k": desc(k), "v": desc(v)}
    for name in ("max_seqlen_q", "max_seqlen_k", "num_splits", "causal", "window_size", "softcap", "fa_version"):
        rec[name] = desc(kw.get(name))
    for name in ("seqused_k", "cu_seqlens_q", "block_table"):
        t = kw.get(name)
        rec[name] = desc(t)
        if isinstance(t, torch.Tensor) and t.numel() <= 8:
            rec[name + "_values"] = t.tolist()
    for name in ("scheduler_metadata", "dynamic_causal", "s_aux", "mask_mod", "aux_tensors"):
        rec[name] = None if kw.get(name) is None else desc(kw.get(name))
    key = json.dumps({x: rec[x] for x in ("q", "k", "num_splits", "max_seqlen_q", "window_size")}, sort_keys=True)
    if rec["max_seqlen_q"] == 1 and key not in seen:  # decode calls only
        seen[key] = rec
        print("ATTN_CALL", json.dumps(rec), flush=True)
    return orig_call(self, q=q, k=k, v=v, **kw)


kernel_cls.__call__ = logged_call

llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
          gpu_memory_utilization=args.gpu_mem_util, enable_prefix_caching=False, enforce_eager=True)
filler = "The quick brown fox jumps over the lazy dog. "
text = filler * (args.context // 10) + "Summarize the text."
sp = SamplingParams(temperature=0.0, max_tokens=4, min_tokens=4, ignore_eos=True)
llm.chat([[{"role": "user", "content": text}]], sp, use_tqdm=False)
with open(args.out, "w") as f:
    json.dump(list(seen.values()), f, indent=1)
print("PROBE_DONE", len(seen))
