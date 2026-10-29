#!/usr/bin/env python3
"""Roofline-style theoretical limits for Gemma 4 (and baselines) on H100 NVL.

Everything is derived from the checkpoint itself:
  * safetensors headers -> exact bytes per weight category
  * config.json         -> layer types, sliding window, KV heads, K=V, KV sharing

Decode (one step, batch B, context L):
  bytes  = W_dense + W_lm_head + W_experts * f_experts(B) + B * KV_read(L)
  flops  = B * (2 * N_active + attention_flops(L))
  t_step >= max(bytes / BW, flops / PEAK)        (roofline lower bound)
Prefill (prompt P): flops = 2 * P * N_body_active + causal attention, TTFT >= flops / PEAK.
"""
from __future__ import annotations

import glob
import json
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "theory.json"
HUB = Path.home() / ".cache/huggingface/hub"

# NVIDIA H100 NVL product brief (per GPU). Tensor-core numbers are *dense*
# (datasheet "with sparsity" figures are 2x these).
HW = {
    "name": "NVIDIA H100 NVL (PCIe, 94 GB HBM3)",
    "sms": 132,
    "max_sm_clock_mhz": 1785,
    "hbm_gib": 94,
    "hbm_bw_tbs": 3.9,
    "bf16_tflops_dense": 835.0,
    "fp8_tflops_dense": 1671.0,
    # 132 SM * 4 TC * 1024 dense BF16 FLOP/clk * 1.785 GHz: upper bound at max boost
    "bf16_tflops_at_max_clock": 132 * 4 * 1024 * 1.785e9 / 1e12,
    "l2_mib": 60,
    "tdp_w": 400,
}

MODELS = {
    "E2B": ("google/gemma-4-E2B-it", "google/gemma-4-E2B-it-assistant"),
    "E4B": ("google/gemma-4-E4B-it", "google/gemma-4-E4B-it-assistant"),
    "12B": ("google/gemma-4-12B-it", "google/gemma-4-12B-it-assistant"),
    "26B_A4B": ("google/gemma-4-26B-A4B-it", "google/gemma-4-26B-A4B-it-assistant"),
    "31B": ("google/gemma-4-31B-it", "google/gemma-4-31B-it-assistant"),
    "Qwen3_30B_A3B": ("Qwen/Qwen3-30B-A3B-Instruct-2507", None),
    "Qwen3_32B": ("Qwen/Qwen3-32B", None),
}
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "F8_E4M3": 1}

# `attention_k_eq_v` (global layers of 12B/26B/31B) drops v_proj: V is built from the K
# projection. But V = v_norm(k_proj(x)) without RoPE while K = rope(k_norm(k_proj(x))), so
# both HF transformers and vLLM cache K *and* V. Count what the implementations store/read.
K_EQ_V_STORED_ONCE = False


def snapshot(repo: str) -> Path | None:
    snaps = sorted(glob.glob(str(HUB / f"models--{repo.replace('/', '--')}/snapshots/*")))
    return Path(snaps[-1]) if snaps else None


def tensors(snap: Path):
    for f in sorted(snap.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            numel = 1
            for s in v["shape"]:
                numel *= s
            yield k, v["shape"], numel * DTYPE_BYTES[v["dtype"]]


def categorize(snap: Path) -> dict[str, float]:
    """Bytes per category. Vision/audio towers are irrelevant for text serving."""
    cat = {"embed": 0.0, "lm_head": 0.0, "ple_table": 0.0, "experts": 0.0,
           "body": 0.0, "multimodal": 0.0}
    for k, _shape, nb in tensors(snap):
        if re.search(r"(vision|audio)_tower|embed_(vision|audio)", k):
            cat["multimodal"] += nb
        elif "embed_tokens_per_layer" in k:
            cat["ple_table"] += nb
        elif k.endswith("embed_tokens.weight"):
            cat["embed"] += nb
        elif k.startswith("lm_head"):
            cat["lm_head"] += nb
        elif ".experts." in k or k.endswith((".experts.gate_up_proj", ".experts.down_proj")):
            cat["experts"] += nb
        else:
            cat["body"] += nb
    return cat


def text_cfg(snap: Path) -> dict:
    c = json.loads((snap / "config.json").read_text())
    return c.get("text_config", c)


def kv_geometry(tc: dict) -> list[dict]:
    """Per-layer attention geometry, incl. which layers own a KV cache."""
    L = tc["num_hidden_layers"]
    types = tc.get("layer_types") or ["full_attention"] * L
    shared = tc.get("num_kv_shared_layers") or 0
    win = tc.get("sliding_window")
    layers = []
    for i, t in enumerate(types):
        glob_ = t == "full_attention"
        if glob_ and tc.get("global_head_dim"):
            hd = tc["global_head_dim"]
            kvh = tc.get("num_global_key_value_heads") or tc["num_key_value_heads"]
            kv_mult = 1 if (tc.get("attention_k_eq_v") and K_EQ_V_STORED_ONCE) else 2
        else:
            hd = tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"]
            kvh = tc["num_key_value_heads"]
            kv_mult = 2
        layers.append({
            "global": glob_,
            "window": None if glob_ else win,
            "head_dim": hd,
            "q_heads": tc["num_attention_heads"],
            "kv_bytes_per_tok": kvh * hd * kv_mult * 2,  # BF16
            "owns_kv": i < L - shared,
        })
    return layers


def kv_bytes_stored(layers, ctx: int) -> float:
    return sum(l["kv_bytes_per_tok"] * (min(ctx, l["window"]) if l["window"] else ctx)
               for l in layers if l["owns_kv"])


def kv_bytes_read(layers, ctx: int) -> float:
    # every layer (KV-shared ones too) attends over its visible span once per step
    return sum(l["kv_bytes_per_tok"] * (min(ctx, l["window"]) if l["window"] else ctx)
               for l in layers)


def attn_flops_decode(layers, ctx: int) -> float:
    return sum(4 * l["q_heads"] * l["head_dim"] * (min(ctx, l["window"]) if l["window"] else ctx)
               for l in layers)


def attn_flops_prefill(layers, P: int) -> float:
    tot = 0.0
    for l in layers:
        w = l["window"]
        # sum_{p=1..P} min(p, w)
        span = P * (P + 1) / 2 if not w or P <= w else w * (w + 1) / 2 + (P - w) * w
        tot += 4 * l["q_heads"] * l["head_dim"] * span
    return tot


def draft_bytes(repo: str | None) -> float | None:
    if not repo:
        return None
    snap = snapshot(repo)
    if not snap:
        return None
    c = json.loads((snap / "config.json").read_text())
    tc = c.get("text_config", c)
    total, emb = 0.0, 0.0
    for k, shape, nb in tensors(snap):
        if k.endswith("embed_tokens.weight"):
            emb = nb
        else:
            total += nb
    if c.get("use_ordered_embeddings") and c.get("num_centroids"):
        # centroid LM head: score centroids, then only top-k clusters' rows
        frac = c["centroid_intermediate_top_k"] / c["num_centroids"]
        total += emb * frac
    else:
        total += emb
    return total


def expert_frac(tc: dict, batch: int) -> float:
    E = tc.get("num_experts")
    k = tc.get("top_k_experts") or tc.get("num_experts_per_tok")
    if not E:
        return 1.0
    # expected fraction of distinct experts touched by `batch` independent tokens
    return 1 - (1 - k / E) ** batch


def model_theory(tag: str, repo: str, draft_repo: str | None) -> dict | None:
    snap = snapshot(repo)
    if not snap or not list(snap.glob("*.safetensors")):
        return None
    tc = text_cfg(snap)
    cat = categorize(snap)
    lm_head = cat["lm_head"] or cat["embed"]  # tied embeddings -> full read for logits
    layers = kv_geometry(tc)
    bw = HW["hbm_bw_tbs"] * 1e12
    peak = HW["bf16_tflops_dense"] * 1e12
    n_body = cat["body"] / 2
    n_exp_active = cat["experts"] / 2 * expert_frac(tc, 1)
    n_active = n_body + n_exp_active + lm_head / 2

    def decode(batch: int, ctx: int) -> dict:
        wbytes = cat["body"] + lm_head + cat["experts"] * expert_frac(tc, batch)
        kv = batch * kv_bytes_read(layers, ctx)
        flops = batch * (2 * n_active + attn_flops_decode(layers, ctx))
        t_mem, t_cmp = (wbytes + kv) / bw, flops / peak
        t = max(t_mem, t_cmp)
        return {"batch": batch, "ctx": ctx, "weight_bytes": wbytes, "kv_bytes": kv,
                "flops": flops, "t_mem_ms": t_mem * 1e3, "t_compute_ms": t_cmp * 1e3,
                "bound": "memory" if t_mem >= t_cmp else "compute",
                "tpot_floor_ms": t * 1e3, "tok_s_per_seq": 1 / t, "tok_s_total": batch / t,
                "arith_intensity": flops / (wbytes + kv)}

    def prefill(P: int) -> dict:
        flops = 2 * P * (n_body + cat["experts"] / 2 * expert_frac(tc, 1)) \
            + attn_flops_prefill(layers, P) + 2 * lm_head / 2
        return {"P": P, "flops": flops, "ttft_floor_ms": flops / peak * 1e3}

    db = draft_bytes(draft_repo)
    return {
        "tag": tag, "repo": repo,
        "bytes": cat,
        "params_b": {k: v / 2e9 for k, v in cat.items()},
        "n_active_decode_b": n_active / 1e9,
        "n_body_b": n_body / 1e9,
        "layers": tc["num_hidden_layers"],
        "num_global_layers": sum(l["global"] for l in layers),
        "num_kv_owning_layers": sum(l["owns_kv"] for l in layers),
        "sliding_window": tc.get("sliding_window"),
        "kv_bytes_per_token_full_ctx": {c: kv_bytes_stored(layers, c) / c
                                        for c in (1024, 4096, 32768, 131072)},
        "kv_stored_gib": {c: kv_bytes_stored(layers, c) / 2**30 for c in (4096, 32768, 131072)},
        "kv_stored_gib_no_window": {
            c: sum(l["kv_bytes_per_tok"] * c for l in layers if l["owns_kv"]) / 2**30
            for c in (4096, 32768)},
        "draft_bytes_per_step": db,
        "draft_to_target_bytes": db / (cat["body"] + lm_head + cat["experts"] * expert_frac(tc, 1))
        if db else None,
        "decode": [decode(b, c) for b in (1, 2, 4, 8, 16, 32, 64, 128, 256)
                   for c in (1024, 2048, 4096, 8192, 16384, 32768)],
        "prefill": [prefill(P) for P in (512, 1024, 2048, 4096, 8192, 16384, 32768)],
        "experts": {"num": tc.get("num_experts"),
                    "top_k": tc.get("top_k_experts") or tc.get("num_experts_per_tok"),
                    "frac_touched": {b: expert_frac(tc, b) for b in (1, 4, 16, 64)}},
    }


def main() -> None:
    out = {"hardware": HW, "models": []}
    for tag, (repo, draft) in MODELS.items():
        t = model_theory(tag, repo, draft)
        if t is None:
            print(f"skip {tag}: weights not downloaded")
            continue
        out["models"].append(t)
        d1 = next(x for x in t["decode"] if x["batch"] == 1 and x["ctx"] == 2048)
        print(f"{tag:<14} active={t['n_active_decode_b']:6.2f}B  "
              f"bytes/step(B=1)={d1['weight_bytes']/1e9:6.2f}GB  "
              f"TPOT floor={d1['tpot_floor_ms']:6.2f}ms ({d1['tok_s_per_seq']:6.0f} tok/s)  "
              f"KV@32k={t['kv_stored_gib'][32768]:5.2f}GiB (no-window "
              f"{t['kv_stored_gib_no_window'][32768]:5.2f})  "
              f"draft/target={t['draft_to_target_bytes'] or 0:.3f}")
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
