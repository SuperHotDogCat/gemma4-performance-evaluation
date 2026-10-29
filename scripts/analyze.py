#!/usr/bin/env python3
"""Turn data/serve/*.json (+ theory, microbench, nsys) into report/figures/*.{pdf,png} and report/tables/*.tex."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theory as th  # noqa: E402

DATA, FIG, TAB = ROOT / "data", ROOT / "report" / "figures", ROOT / "report" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# fixed categorical order. The report is printed in monochrome, so a model is identified by
# marker shape (+ fill) and line style instead of hue: filled = dense, hollow = MoE,
# solid line = Gemma 4, dash-dot = Qwen3.
ORDER = ["E2B", "E4B", "12B", "26B_A4B", "31B", "Qwen3_30B_A3B", "Qwen3_32B"]
LABEL = {"E2B": "Gemma4 E2B", "E4B": "Gemma4 E4B", "12B": "Gemma4 12B",
         "26B_A4B": "Gemma4 26B-A4B", "31B": "Gemma4 31B",
         "Qwen3_30B_A3B": "Qwen3-30B-A3B", "Qwen3_32B": "Qwen3-32B"}
INK, MUTED, GRID = "#000000", "#555555", "#dddddd"
COLOR = {m: (INK if not m.startswith("Qwen") else "#666666") for m in ORDER}
MARKER = dict(zip(ORDER, ["o", "s", "^", "D", "v", "X", "P"]))
HOLLOW = {"26B_A4B", "Qwen3_30B_A3B"}
LS = {m: ("-." if m.startswith("Qwen") else "-") for m in ORDER}
# bar charts: gray level + hatch per model
BAR = dict(zip(ORDER, [("#ffffff", "////"), ("#d0d0d0", ""), ("#ffffff", "...."),
                       ("#808080", ""), ("#202020", ""), ("#ffffff", "xx"), ("#b0b0b0", "\\\\")]))
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.3,
    "lines.markersize": 4, "legend.frameon": False, "figure.dpi": 150, "hatch.linewidth": 0.6,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def sty(m, ls=None, ms=4, **kw):
    """Line/marker kwargs that identify model `m` without color."""
    return dict(color=COLOR[m], marker=MARKER[m], ls=ls or LS[m], ms=ms,
                mfc="white" if m in HOLLOW else COLOR[m], mec=COLOR[m], **kw)


def floor_sty(m):
    """Thin dotted theory line with sparse hollow markers so it can be matched to its model."""
    return dict(color=COLOR[m], ls=":", lw=0.9, marker=MARKER[m], ms=3, mfc="white",
                mec=COLOR[m], markevery=6)


def bar_sty(m):
    fc, h = BAR[m]
    return dict(color=fc, hatch=h, edgecolor=INK, linewidth=0.6)


THEORY = json.loads((DATA / "theory.json").read_text())
HW = THEORY["hardware"]
BW = HW["hbm_bw_tbs"] * 1e12
PEAK = HW["bf16_tflops_dense"] * 1e12
MB = json.loads((DATA / "microbench.json").read_text()) if (DATA / "microbench.json").exists() else None


# ------------------------------------------------------------------ per-model cost model
class Cost:
    """Exact per-request FLOPs / bytes from the checkpoint (same formulas as theory.py)."""

    def __init__(self, tag: str):
        repo, draft = th.MODELS[tag]
        snap = th.snapshot(repo)
        self.tc = th.text_cfg(snap)
        self.cat = th.categorize(snap)
        self.lm = self.cat["lm_head"] or self.cat["embed"]
        self.layers = th.kv_geometry(self.tc)
        self.n_body = self.cat["body"] / 2 + self.cat["experts"] / 2 * th.expert_frac(self.tc, 1)

    def weight_bytes(self, batch=1):
        return self.cat["body"] + self.lm + self.cat["experts"] * th.expert_frac(self.tc, batch)

    def step_bytes(self, ctx, batch=1):
        return self.weight_bytes(batch) + batch * th.kv_bytes_read(self.layers, ctx)

    def prefill_flops(self, P):
        return 2 * P * self.n_body + th.attn_flops_prefill(self.layers, P) + self.lm

    def step_floor_with_prefill(self, ctx, batch, prefill_per_out):
        """Steady-state step time when each output token also carries `prefill_per_out`
        prompt tokens of (chunked) prefill work -- the mixed batch of a busy server."""
        fl = batch * (2 * (self.n_body + self.lm / 2) + th.attn_flops_decode(self.layers, ctx)
                      + prefill_per_out * 2 * self.n_body)
        return max(self.step_bytes(ctx, batch) / BW, fl / PEAK)

    def tpot_floor(self, ctx, batch=1):
        fl = batch * (2 * (self.n_body + self.lm / 2) + th.attn_flops_decode(self.layers, ctx))
        return max(self.step_bytes(ctx, batch) / BW, fl / PEAK)


_COST: dict[str, Cost] = {}


def cost(tag):
    if tag not in _COST:
        _COST[tag] = Cost(tag)
    return _COST[tag]


# ------------------------------------------------------------------ load
def load():
    lat, conc, meta = [], [], {}
    for f in sorted((DATA / "serve").glob("*.json")):
        d = json.loads(f.read_text())
        tag = d["tag"]
        base = next((m for m in ORDER if tag == m or tag.startswith(m + "_")), None)
        if base is None:
            continue
        variant = tag[len(base):].lstrip("_") or "base"
        meta[tag] = d
        for r in d.get("latency", []):
            lat.append({**{k: v for k, v in r.items() if k != "spec"}, "tag": tag, "model": base,
                        "variant": variant,
                        "al": (r.get("spec") or {}).get("mean_acceptance_length"),
                        "acc_rate": (r.get("spec") or {}).get("acceptance_rate"),
                        "acc_pos": (r.get("spec") or {}).get("acc_per_pos")})
        for r in d.get("concurrency", []):
            conc.append({**{k: v for k, v in r.items() if k != "spec"}, "tag": tag, "model": base,
                         "variant": variant,
                         "al": (r.get("spec") or {}).get("mean_acceptance_length")})
    return pd.DataFrame(lat), pd.DataFrame(conc), meta


def models_in(df):
    return [m for m in ORDER if m in set(df["model"])]


def fig_legend(fig, ncol=4):
    """One legend under the panels instead of a crowded in-plot box."""
    h, l = [], []
    for ax in fig.axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
        if ax.get_legend():
            ax.get_legend().remove()
    fig.tight_layout()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=ncol, fontsize=7)


def save(fig, name):
    from matplotlib.ticker import FuncFormatter, NullFormatter
    for ax in fig.axes:
        for axis, sc in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if sc == "log":
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                axis.set_minor_formatter(NullFormatter())
    if not fig.legends:
        fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ figures
# One figure = one claim of the report: each figure only shows the models the claim is about,
# and lines are labelled directly where possible (no colour to match against a legend).
DENSE_PAIR, MOE_PAIR = ("31B", "Qwen3_32B"), ("26B_A4B", "Qwen3_30B_A3B")
XT_CONC = ([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])


def end_label(ax, x, y, text, dx=4, dy=0, ha="left"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                va="center", ha=ha, color=INK)


def base_len_sweep(lat):
    """Baseline, concurrency 1, Synthetic + Babilong, one row per (model, workload)."""
    b = lat[(lat.variant == "base") & (lat.kind != "swe_bench_lite")]
    return b.groupby(["model", "workload_id"], as_index=False).agg(
        prompt_tokens=("prompt_tokens", "first"), output_tokens=("output_tokens", "first"),
        ttft_s=("ttft_s", "median"), tpot_s=("tpot_s", "median"))


def fig_prefill_rate(lat, models=("12B", "31B", "Qwen3_32B", "26B_A4B", "Qwen3_30B_A3B")):
    """Claim: large dense models reach ~80% of the GEMM peak this GPU can actually deliver;
    MoE models (small per-expert GEMMs) stay well below."""
    b = base_len_sweep(lat)
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    for m in [m for m in models if m in set(b.model)]:
        s = b[b.model == m].sort_values("prompt_tokens")
        eff = [cost(m).prefill_flops(p) / t / 1e12 for p, t in zip(s.prompt_tokens, s.ttft_s)]
        ax.plot(s.prompt_tokens, eff, label=LABEL[m], **sty(m, ms=3))
    ax.axhline(PEAK / 1e12, color=MUTED, lw=1, ls="--")
    ax.text(300, PEAK / 1e12 - 15, "datasheet BF16 peak 835", color=MUTED, fontsize=7, va="top")
    if MB:
        ax.axhline(MB["bf16_tflops_best"], color=MUTED, lw=1, ls=":")
        ax.text(300, MB["bf16_tflops_best"] + 15, f"measured GEMM peak {MB['bf16_tflops_best']:.0f}",
                color=MUTED, fontsize=7, va="bottom")
    ax.set(xscale="log", xlabel="prompt tokens", ylabel="achieved TFLOPS", ylim=(0, 900),
           title="Prefill FLOP rate (filled: dense, hollow: MoE)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, "prefill_rate")


def fig_prefill_hybrid(lat):
    """Claim: the TTFT advantage over Qwen3 appears only once the prompt is well past the window,
    and tracks the FLOPs saved by the sliding-window layers."""
    b = base_len_sweep(lat)
    P = np.geomspace(256, 32768, 40)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
    for ax, (g, q), ttl in ((axes[0], DENSE_PAIR, "31B / Qwen3-32B"), (axes[1], MOE_PAIR, "26B-A4B / Qwen3-30B-A3B")):
        cg, cq = cost(g), cost(q)
        allglob = [{**l, "window": None} for l in cg.layers]

        def fl_allglob(p):
            return cg.prefill_flops(p) - th.attn_flops_prefill(cg.layers, p) + th.attn_flops_prefill(allglob, p)

        j = b[b.model == g].merge(b[b.model == q], on="workload_id", suffixes=("", "_q"))
        ax.plot(j.prompt_tokens, j.ttft_s / j.ttft_s_q, label="measured TTFT ratio", **sty(g, ls="none", ms=3))
        ax.plot(P, [cg.prefill_flops(p) / cq.prefill_flops(p) for p in P], color=INK, lw=1.2,
                label="FLOPs ratio (theory)")
        ax.plot(P, [fl_allglob(p) / cq.prefill_flops(p) for p in P], color=INK, ls=":", lw=1.0,
                label="FLOPs ratio if every Gemma layer were global")
        ax.axhline(1, color=MUTED, lw=0.8)
        ax.axvline(1024, color=MUTED, ls="--", lw=0.8)
        ax.text(1100, 1.24, "window 1024", fontsize=7, color=MUTED)
        ax.set(xscale="log", xlabel="prompt tokens", title=ttl, ylim=(0.45, 1.3))
    axes[0].set_ylabel("Gemma 4 / Qwen3  (<1: Gemma faster)")
    axes[0].legend(fontsize=7, loc="lower left")
    save(fig, "prefill_hybrid")


def short_ctx_point(lat, m):
    s = lat[(lat.model == m) & (lat.variant == "base") & (lat.kind == "synthetic")
            & (lat.prompt_tokens < 2000)]
    ctx = (s.prompt_tokens + 64).median()
    tp = s.tpot_s.median()
    return cost(m).step_bytes(ctx), tp


def fig_decode_bw(lat):
    """Claim: the more bytes one decode step reads, the closer to the HBM roofline."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for m in models_in(lat):
        by, tp = short_ctx_point(lat, m)
        u = by / tp / BW * 100
        ax.plot([by / 1e9], [u], **sty(m, ls="none", ms=6))
        left = m in ("26B_A4B", "Qwen3_30B_A3B")
        end_label(ax, by / 1e9, u, LABEL[m].replace("Gemma4 ", ""), dx=-7 if left else 7,
                  dy=4 if m == "26B_A4B" else (-4 if m == "Qwen3_30B_A3B" else 0),
                  ha="right" if left else "left")
    if MB:
        ax.axhline(MB["read_bw_tbs"] / HW["hbm_bw_tbs"] * 100, color=MUTED, ls="--", lw=1)
        ax.text(3.2, MB["read_bw_tbs"] / HW["hbm_bw_tbs"] * 100 + 1.5, "measured stream-read BW",
                color=MUTED, fontsize=7)
    ax.set(xscale="log", xlim=(3, 200), ylim=(0, 100), xlabel="bytes read per decode step [GB]",
           ylabel="HBM utilization [% of 3.9 TB/s]", title="Short-context decode (concurrency 1)")
    ax.set_xticks([4, 8, 16, 32, 64])
    save(fig, "decode_bw")


def fig_decode_gap():
    """Claim: the gap between measured TPOT and the bandwidth floor is made of many small
    per-kernel costs (under-filled GEMMs, small norm/elementwise kernels, attention launches, idle)."""
    if not MB:
        return
    spec = [("E2B", "E2B_b1"), ("26B_A4B", "26B_A4B_b1"), ("Qwen3_30B_A3B", "Qwen3_30B_A3B_b1"),
            ("12B", "12B_b1"), ("31B", "31B_b1")]
    parts = [("ideal", "weights / measured HBM BW", "#ffffff", ""),
             ("gemm_x", "GEMM+MoE beyond ideal", "#303030", ""),
             ("attention", "attention", "#a0a0a0", ""),
             ("norm/elementwise", "norm / elementwise", "#dcdcdc", "...."),
             ("rest", "sampling / other", "#000000", ""),
             ("idle", "GPU idle", "#ffffff", "////")]
    rows = []
    for m, tag in spec:
        f = DATA / "nsys" / f"{tag}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        steps = d["run"]["tokens"] if d.get("run") else 256
        ms = {k: v / steps for k, v in d["kernel_time_by_cat_ms"].items()}
        step = d["span_ms"] / steps
        ideal = cost(m).weight_bytes(1) / (MB["read_bw_tbs"] * 1e12) * 1e3
        gm = ms.get("gemm", 0) + ms.get("moe", 0)
        rows.append((m, {"ideal": ideal, "gemm_x": max(gm - ideal, 0), "attention": ms.get("attention", 0),
                         "norm/elementwise": ms.get("norm/elementwise", 0),
                         "rest": ms.get("sampling/logits", 0) + ms.get("other", 0),
                         "idle": step - d["gpu_busy_ms"] / steps}))
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for i, (m, v) in enumerate(rows[::-1]):
        left = 0.0
        for k, lab, fc, h in parts:
            ax.barh(i, v[k], left=left, color=fc, hatch=h, edgecolor=INK, linewidth=0.5, height=0.62,
                    label=lab if i == 0 else None)
            left += v[k]
        ax.text(left + 0.15, i, f"{left:.1f} ms  (ideal {v['ideal']:.1f})", va="center", fontsize=7)
    ax.set_yticks(range(len(rows)), [LABEL[m] for m, _ in rows[::-1]], fontsize=7)
    ax.set(xlabel="time per decode step [ms] (batch 1, short context)", xlim=(0, 28),
           title="Where the gap to the bandwidth floor comes from")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.0))
    save(fig, "decode_gap")
    for m, v in rows:
        print("gap", m, {k: round(x, 2) for k, x in v.items()})


def fig_decode_long(lat):
    """Claim: Gemma's TPOT grows with context far beyond its KV-read floor; Qwen3's does not."""
    b = base_len_sweep(lat)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=False)
    for ax, pair, ttl in ((axes[0], DENSE_PAIR, "Dense ~31B"), (axes[1], MOE_PAIR, "MoE ~3-4B active")):
        for m in pair:
            s = b[b.model == m].sort_values("prompt_tokens")
            ctx = s.prompt_tokens + s.output_tokens / 2
            ax.plot(s.prompt_tokens, s.tpot_s * 1e3, label=LABEL[m], **sty(m, ms=3))
            ax.plot(s.prompt_tokens, [cost(m).tpot_floor(x) * 1e3 for x in ctx], **floor_sty(m))
        ax.set(xscale="log", xlabel="prompt tokens", ylabel="TPOT [ms]", ylim=(0, None),
               title=f"{ttl} (dotted: HBM floor)")
        ax.legend(fontsize=7, loc="upper left")
    save(fig, "decode_long")


DRAFT_BYTES = {t["tag"]: t["draft_bytes_per_step"] for t in THEORY["models"]}


def mtp_pairs(lat):
    """Join base and spec runs per (model, workload, repeat)."""
    key = ["model", "workload_id", "repeat"]
    base = lat[lat.variant == "base"][key + ["tpot_s", "e2e_s", "ttft_s", "output_tokens"]]
    spec = lat[lat.variant != "base"]
    j = spec.merge(base, on=key, suffixes=("", "_base"))
    j["speedup_tpot"] = j.tpot_s_base / j.tpot_s
    j["speedup_e2e"] = j.e2e_s_base / j.e2e_s
    return j


def spec_step_cost(tag, gamma, ctx=2048):
    """Bandwidth model: (target verify of gamma+1 tokens + gamma draft steps) / one plain step."""
    c = cost(tag)
    verify = c.step_bytes(ctx, gamma + 1)
    return (verify + gamma * (DRAFT_BYTES.get(tag) or 0)) / c.step_bytes(ctx, 1)


def bar_labels(ax, bars, fmt):
    for r in bars:
        ax.annotate(fmt.format(r.get_height()), (r.get_x() + r.get_width() / 2, r.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)


def fig_mtp(lat):
    j = mtp_pairs(lat)
    j4 = j[(j.variant == "mtp4") & (j.kind != "babilong")]
    if j4.empty:
        return
    # Claim: AL is similar across models, but the speedup is not
    ms = models_in(j4)
    g = j4.groupby("model").agg(al=("al", "mean"), sp=("speedup_tpot", "median")).reindex(ms)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.7))
    x = np.arange(len(ms))
    names = [LABEL[m].replace("Gemma4 ", "") for m in ms]
    for ax, col, ttl, yl, fmt in ((axes[0], "al", "Mean acceptance length (max 5)", (1, 5), "{:.2f}"),
                                  (axes[1], "sp", "Decode speedup (TPOT base / MTP)", (0, 3.6), "{:.2f}x")):
        bars = ax.bar(x, g[col], 0.6, color="#b0b0b0", edgecolor=INK, linewidth=0.6)
        bars[ms.index("26B_A4B")].set_hatch("////") if "26B_A4B" in ms else None
        bar_labels(ax, bars, fmt)
        ax.set_xticks(x, names)
        ax.set(title=ttl, ylim=yl)
        ax.grid(axis="x", visible=False)
    axes[1].axhline(1, color=MUTED, lw=1)
    save(fig, "mtp_speedup")

    # Claim: MTP is content-independent, n-gram is not (31B, gamma=4)
    kinds = [("synthetic", "Synthetic\n(free text)"), ("swe_bench_lite", "SWE-bench Lite\n(copy-heavy)"),
             ("babilong", "Babilong*\n(repetitive)")]
    s = j[(j.model == "31B") & j.variant.isin(["mtp4", "ngram4"])]
    if s.variant.nunique() == 2:
        fig, ax = plt.subplots(figsize=(4.8, 2.8))
        x = np.arange(len(kinds))
        for k, (v, lab, fc, h) in enumerate((("mtp4", "MTP drafter", "#404040", ""),
                                             ("ngram4", "n-gram (prompt lookup)", "#ffffff", "////"))):
            sp = s[s.variant == v].groupby("kind").speedup_tpot.median()
            bars = ax.bar(x + (k - 0.5) * 0.36, [sp.get(kk, np.nan) for kk, _ in kinds], 0.36,
                          color=fc, hatch=h, edgecolor=INK, linewidth=0.6, label=lab)
            bar_labels(ax, bars, "{:.2f}x")
        ax.axhline(1, color=MUTED, lw=1)
        ax.set_xticks(x, [k[1] for k in kinds])
        ax.set(ylabel="TPOT speedup", ylim=(0, 4.2), title="Gemma4 31B, $\\gamma$=4: MTP vs n-gram")
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=7, loc="upper left")
        save(fig, "mtp_vs_ngram")

    # SWE-bench genre view (appendix)
    s = j4[j4.kind == "swe_bench_lite"]
    if not s.empty:
        g = s.groupby(["genre", "model"]).agg(al=("al", "mean")).reset_index()
        genres = sorted(g.genre.unique())
        w = 0.8 / len(ms)
        fig, ax = plt.subplots(figsize=(7.6, 2.8))
        for i, m in enumerate(models_in(g)):
            gg = g[g.model == m].set_index("genre").reindex(genres)
            ax.bar(np.arange(len(genres)) + (i - (len(ms) - 1) / 2) * w, gg.al, w * 0.9,
                   label=LABEL[m], **bar_sty(m))
        ax.set_xticks(range(len(genres)), genres, rotation=20)
        ax.grid(axis="x", visible=False)
        ax.set(ylabel="mean acceptance length", ylim=(1, 5),
               title="MTP acceptance by SWE-bench Lite repository (natural greedy generation)")
        ax.legend(fontsize=7, ncol=5, loc="upper left")
        save(fig, "mtp_swe_genre")


def spec_table(lat):
    j = mtp_pairs(lat)
    j = j[j.variant.str.startswith("mtp") & (j.kind != "babilong")]
    g = j.groupby(["model", "variant"]).agg(al=("al", "mean"), sp=("speedup_tpot", "median")).reset_index()
    g["gamma"] = g.variant.str[3:].astype(int)
    g["c_pred"] = [spec_step_cost(m, k) for m, k in zip(g.model, g.gamma)]
    g["c_meas"] = g.al / g.sp
    g["sp_pred"] = g.al / g.c_pred
    return j, g


def fig_spec_model(lat):
    j, g = spec_table(lat)
    if g.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    # left: gamma=4 only, one labelled point per model
    g4 = g[g.gamma == 4]
    for _, r in g4.iterrows():
        axes[0].plot([r.sp_pred], [r.sp], **sty(r.model, ls="none", ms=6))
        dy = {"E2B": 5, "12B": -6, "E4B": 0}.get(r.model, 0)
        end_label(axes[0], r.sp_pred, r.sp, LABEL[r.model].replace("Gemma4 ", ""), dx=-7 if r.model == "E2B" else 7,
                  dy=dy, ha="right" if r.model == "E2B" else "left")
    lim = [1.0, 3.6]
    axes[0].plot(lim, lim, color=MUTED, lw=1)
    axes[0].text(1.6, 1.75, "prediction = measurement", rotation=40, fontsize=7, color=MUTED)
    axes[0].set(xlabel="predicted speedup  AL / $c_{pred}$", ylabel="measured TPOT speedup",
                title="Bandwidth model vs measurement ($\\gamma$=4)", xlim=lim, ylim=lim)
    # right: step cost vs gamma, only models with a gamma sweep
    for m in models_in(g):
        s_ = g[g.model == m].sort_values("gamma")
        if len(s_) < 2:
            continue
        axes[1].plot(s_.gamma, s_.c_meas, label=LABEL[m], **sty(m))
        axes[1].plot(s_.gamma, s_.c_pred, **{**floor_sty(m), "markevery": 1})
    axes[1].set(xlabel="$\\gamma$", ylabel="cost of one spec step / plain step",
                title="Step cost $c$ (solid: measured, dotted: model)")
    axes[1].set_xticks([1, 2, 4, 8])
    axes[1].legend(fontsize=7, loc="upper left")
    save(fig, "mtp_cost_model")
    g.round(3).to_csv(DATA / "spec_cost_model.csv", index=False)
    print(g.round(2).to_string())


def fig_gamma(lat):
    j, _ = spec_table(lat)
    if j.empty:
        return
    j = j.assign(gamma=j.variant.str[3:].astype(int))
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))
    for m in models_in(j):
        s = j[j.model == m]
        if s.gamma.nunique() < 2:
            continue
        g = s.groupby("gamma").agg(al=("al", "mean"), sp=("speedup_tpot", "median")).reset_index()
        axes[0].plot(g.gamma, g.al, label=LABEL[m], **sty(m))
        axes[1].plot(g.gamma, g.sp, **sty(m))
        gmax = s[s.gamma == s.gamma.max()]
        pos = np.nanmean(np.array([p for p in gmax.acc_pos if p], dtype=float), axis=0)
        axes[2].plot(np.arange(1, len(pos) + 1), pos, **sty(m))
    axes[0].plot([1, 8], [2, 9], ":", color=MUTED, lw=1)
    axes[0].set(xlabel="$\\gamma$ (draft tokens)", ylabel="acceptance length", title="Acceptance length")
    axes[1].axhline(1, color=MUTED, lw=1)
    axes[1].set(xlabel="$\\gamma$", ylabel="TPOT speedup", title="Decode speedup")
    axes[2].set(xlabel="draft position", ylabel="P(accepted)", title="Per-position acceptance",
                ylim=(0, 1))
    fig_legend(fig)
    save(fig, "mtp_gamma")


def conc_base(conc, m):
    return conc[(conc.model == m) & (conc.variant == "base")].sort_values("concurrency")


def fig_concurrency(conc):
    if conc.empty:
        return
    # (1) Claim: batching amortizes the weight read, and decode stays below the ridge point
    ms = [m for m in ("E2B", "12B", "31B", "26B_A4B", "Qwen3_30B_A3B") if m in set(conc.model)]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    ai = np.geomspace(0.5, 1000, 100)
    axes[1].plot(ai, np.minimum(ai * BW, PEAK) / 1e12, color=INK, lw=1.2)
    axes[1].axvline(PEAK / BW, color=MUTED, ls=":", lw=1)
    axes[1].text(PEAK / BW * 1.08, 2, "ridge\n214", fontsize=7, color=MUTED)
    for m in ms:
        c, s = cost(m), conc_base(conc, m)
        axes[0].plot(s.concurrency, s.output_tok_s, label=LABEL[m], **sty(m, ms=3))
        ctx = s.mean_prompt_tokens + 128
        fl = np.array([B * (2 * (c.n_body + c.lm / 2) + th.attn_flops_decode(c.layers, x))
                       for B, x in zip(s.concurrency, ctx)])
        by = np.array([c.step_bytes(x, B) for B, x in zip(s.concurrency, ctx)])
        axes[1].plot(fl / by, fl / s.median_tpot_s.values / 1e12, **sty(m, ms=3))
    axes[0].set(xscale="log", yscale="log", xlabel="concurrency", ylabel="output tok/s",
                title="Throughput")
    axes[0].set_xticks(*XT_CONC)
    axes[0].legend(fontsize=7)
    axes[1].set(xscale="log", yscale="log", xlabel="arithmetic intensity [FLOP/byte]",
                ylabel="achieved TFLOPS", title="Decode roofline (concurrency 1$\\to$64)")
    save(fig, "conc_scaling")

    # (2) Claim: at high load the prefill mixed into each step (chunked prefill) drives TPOT up (12B)
    if "12B" in set(conc.model) and MB:
        c, s = cost("12B"), conc_base(conc, "12B")
        P = s.mean_prompt_tokens.mean()
        ctx, B = P + 128, s.concurrency.values
        t_mem = np.array([c.step_bytes(ctx, b) / BW for b in B])
        t_cmp = np.array([b * (2 * (c.n_body + c.lm / 2) + th.attn_flops_decode(c.layers, ctx)
                               + P / 256 * 2 * c.n_body) for b in B]) / (MB["bf16_tflops_best"] * 1e12)
        fig, ax = plt.subplots(figsize=(4.6, 2.8))
        ax.plot(B, s.median_tpot_s * 1e3, label="measured", **sty("12B"))
        ax.plot(B, [c.tpot_floor(ctx, b) * 1e3 for b in B], color=INK, ls=":", lw=1,
                label="roofline floor (decode only)")
        ax.plot(B, (t_mem + t_cmp) * 1e3, color=INK, ls="--", lw=1,
                label="HBM read + prefill compute, no overlap")
        ax.set(xscale="log", xlabel="concurrency", ylabel="median TPOT [ms]", ylim=(0, None),
               title="Gemma4 12B under load")
        ax.set_xticks(*XT_CONC)
        ax.legend(fontsize=7, loc="upper left")
        save(fig, "conc_prefill")

    # (3) Claim: MoE reads more experts as the batch grows ("3.8B active" only at B=1)
    ms = [m for m in ("26B_A4B", "Qwen3_30B_A3B", "12B") if m in set(conc.model)]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    Bs = np.array([1, 2, 4, 8, 16, 32, 64])
    for m in ms:
        c, s = cost(m), conc_base(conc, m)
        P = s.mean_prompt_tokens.mean()
        axes[0].plot(Bs, [c.weight_bytes(b) / 1e9 for b in Bs], label=LABEL[m], **sty(m))
        axes[1].plot(s.concurrency, s.median_tpot_s * 1e3, **sty(m))
    axes[0].set(xscale="log", xlabel="concurrency $B$", ylabel="weight bytes per step [GB]",
                title="Weights read per decode step (theory)", ylim=(0, None))
    axes[1].set(xscale="log", xlabel="concurrency", ylabel="median TPOT [ms]",
                title="Measured TPOT", ylim=(0, None))
    for ax in axes:
        ax.set_xticks(*XT_CONC)
    axes[0].legend(fontsize=7)
    save(fig, "conc_moe")

    # (4) Claim: 31B saturates its KV pool, Qwen3-32B does not
    ms = [m for m in DENSE_PAIR if m in set(conc.model)]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    for m in ms:
        s = conc_base(conc, m)
        axes[0].plot(s.concurrency, s.output_tok_s, label=LABEL[m], **sty(m))
        axes[1].plot(s.concurrency, s.median_ttft_s, **sty(m))
    axes[0].set(xscale="log", xlabel="concurrency", ylabel="output tok/s", title="Throughput",
                ylim=(0, None))
    axes[1].set(xscale="log", yscale="log", xlabel="concurrency", ylabel="median TTFT [s]",
                title="TTFT (queueing when KV is full)")
    for ax in axes:
        ax.set_xticks(*XT_CONC)
    axes[0].legend(fontsize=7)
    save(fig, "conc_kv")

    # (5) Claim: MTP gain shrinks with load for dense models but grows for MoE
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for m in [m for m in ("E2B", "26B_A4B", "31B") if m in set(conc.model)]:
        b_ = conc_base(conc, m).set_index("concurrency").output_tok_s
        t_ = conc[(conc.model == m) & (conc.variant == "mtp4")].set_index("concurrency").output_tok_s
        r = (t_ / b_).dropna()
        if not r.empty:
            ax.plot(r.index, r.values, label=LABEL[m], **sty(m))
    ax.axhline(1, color=MUTED, lw=1)
    ax.set(xscale="log", xlabel="concurrency", ylabel="throughput MTP / baseline",
           title="MTP ($\\gamma$=4) gain vs load")
    ax.set_xticks(*XT_CONC)
    ax.legend(fontsize=7)
    save(fig, "concurrency_mtp")


def fig_kv(meta):
    """Claim: hybrid attention shrinks long-context KV, but short-context KV is larger than Qwen3."""
    L = np.geomspace(256, 131072, 60)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    for ax, (g, q), ttl in ((axes[0], DENSE_PAIR, "Dense"), (axes[1], MOE_PAIR, "MoE")):
        for m in (g, q):
            c = cost(m)
            kv = np.array([th.kv_bytes_stored(c.layers, int(x)) for x in L])
            ax.plot(L, kv / L / 1024, label=LABEL[m], markevery=8, **sty(m))
        c = cost(g)
        naive = sum(l["kv_bytes_per_tok"] for l in c.layers if l["owns_kv"])
        ax.axhline(naive / 1024, color=MUTED, ls=":", lw=1)
        ax.text(1500, naive / 1024 * 0.93, f"{LABEL[g].split()[-1]}, if every layer were global",
                fontsize=7, color=MUTED, va="top")
        ax.axvline(1024, color=GRID, lw=1, zorder=0)
        ax.set(xscale="log", yscale="log", xlabel="context length [tokens]",
               ylabel="KV per token [KiB]", title=f"{ttl}: average KV bytes per token")
        ax.legend(fontsize=7, loc="lower left")
    save(fig, "kv_footprint")


def fig_babilong_acc(lat):
    b = lat[(lat.kind == "babilong") & (lat.variant == "base")].copy()
    if b.empty:
        return
    b["correct"] = [str(t).lower() in str(x).lower()[:80] for t, x in zip(b.target, b.text)]
    order = ["1k", "2k", "4k", "8k", "16k", "32k"]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for m in models_in(b):
        g = b[b.model == m].groupby("length_config").correct.mean().reindex(order)
        ax.plot(range(len(order)), g * 100, label=LABEL[m], **sty(m, ms=3))
    ax.set_xticks(range(len(order)), order)
    ax.set(xlabel="Babilong context", ylabel="answer contained [%]", ylim=(0, 105),
           title="Babilong qa1-3 (n=6 per length; sanity check)")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, "babilong_accuracy")


NSYS_CATS = ["gemm", "moe", "attention", "sampling/logits", "norm/elementwise", "other"]
# (fill, hatch) per category -- distinguishable in grayscale print
NSYS_STY = dict(zip(NSYS_CATS, [("#303030", ""), ("#ffffff", "////"), ("#a0a0a0", ""),
                                ("#000000", ""), ("#dcdcdc", "...."), ("#ffffff", "xx")]))
NSYS_FIGS = {
    # name: (title, [(profile tag, row label)])
    "nsys_short": ("Decode step, batch 1, short context", [
        ("E2B_b1", "Gemma4 E2B"), ("12B_b1", "Gemma4 12B"), ("26B_A4B_b1", "Gemma4 26B-A4B"),
        ("31B_b1", "Gemma4 31B"), ("Qwen3_30B_A3B_b1", "Qwen3-30B-A3B")]),
    "nsys_long": ("Short vs long context (batch 1)", [
        ("E2B_b1", "E2B, short"), ("E2B_b1_ctx32k", "E2B, ~30k"),
        ("31B_b1", "31B, short"), ("31B_b1_ctx28k", "31B, ~26k")]),
    "nsys_mtp": ("Baseline vs MTP $\\gamma$=4 (batch 1)", [
        ("E2B_b1", "E2B"), ("E2B_mtp4_b1", "E2B +MTP"), ("12B_b1", "12B"), ("12B_mtp4_b1", "12B +MTP"),
        ("26B_A4B_b1", "26B-A4B"), ("26B_A4B_mtp4_b1", "26B-A4B +MTP"),
        ("31B_b1", "31B"), ("31B_mtp4_b1", "31B +MTP")]),
}


def fig_nsys():
    d = DATA / "nsys"
    have = {f.stem: f for f in d.glob("*.json")} if d.exists() else {}
    for name, (title, spec) in NSYS_FIGS.items():
        rows = [(lab, json.loads(have[t].read_text())) for t, lab in spec if t in have][::-1]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(rows) + 1.1))
        used = []
        for i, (lab, r) in enumerate(rows):
            left, busy = 0.0, r["gpu_busy_frac"]
            for c in NSYS_CATS:
                v = r["kernel_time_by_cat_frac"].get(c, 0) * busy * 100
                if v < 0.05:
                    continue
                ax.barh(i, v, left=left, color=NSYS_STY[c][0], hatch=NSYS_STY[c][1], height=0.65,
                        edgecolor=INK, linewidth=0.5)
                used.append(c)
                left += v
            ax.barh(i, 100 - left, left=left, color="white", edgecolor=MUTED, height=0.65, linewidth=0.5)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
        ax.set(xlabel="% of decode wall time (Nsight Systems)", xlim=(0, 100), title=title)
        ax.grid(axis="y", visible=False)
        from matplotlib.patches import Patch
        hs = [Patch(facecolor=NSYS_STY[c][0], hatch=NSYS_STY[c][1], edgecolor=INK, linewidth=0.5, label=c)
              for c in NSYS_CATS if c in used and c != "other"]
        hs.append(Patch(facecolor="white", edgecolor=MUTED, linewidth=0.5, label="GPU idle"))
        fig.tight_layout()
        fig.legend(handles=hs, fontsize=7, ncol=len(hs), loc="upper center", bbox_to_anchor=(0.5, 0.0),
                   handlelength=1.6, columnspacing=1.0)
        save(fig, name)


# ------------------------------------------------------------------ tables
def tex_table(path, header, rows, cols):
    lines = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(x) for x in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / path).write_text("\n".join(lines) + "\n")


def tables(lat, conc, meta):
    if MB:
        # repeat runs (R5 in scripts/run_remeasure.sh) only widen the reported range; figures keep using MB
        runs = [MB] + [json.loads(p.read_text()) for p in sorted((DATA / "remeasure").glob("microbench_*.json"))]
        rng = lambda k, f: "--".join(dict.fromkeys(f"{v:{f}}" for v in (min(r[k] for r in runs), max(r[k] for r in runs))))
        (TAB / "microbench_inline.tex").write_text(
            f"BF16 GEMM 最大 {rng('bf16_tflops_best', '.0f')}\\,TFLOPS，"
            f"ストリーム読み出し {rng('read_bw_tbs', '.2f')}\\,TB/s，コピー {rng('copy_bw_tbs', '.2f')}\\,TB/s（読み+書き，{len(runs)} 回の範囲）")
    else:
        (TAB / "microbench_inline.tex").write_text("（未計測）")
    # theory table
    rows = []
    for t in THEORY["models"]:
        d1 = next(x for x in t["decode"] if x["batch"] == 1 and x["ctx"] == 2048)
        p = next(x for x in t["prefill"] if x["P"] == 4096)
        pb = t["params_b"]
        rows.append([LABEL.get(t["tag"], t["tag"]), f"{t['n_active_decode_b']:.2f}",
                     f"{pb['body'] + pb['experts'] + pb['embed'] + pb['lm_head'] + pb['ple_table']:.1f}",
                     f"{d1['weight_bytes']/1e9:.1f}", f"{d1['tpot_floor_ms']:.2f}",
                     f"{d1['tok_s_per_seq']:.0f}", f"{t['kv_stored_gib']['32768']:.2f}",
                     f"{p['ttft_floor_ms']:.0f}",
                     f"{t['draft_to_target_bytes']*100:.1f}\\%" if t["draft_to_target_bytes"] else "--"])
    tex_table("theory.tex", ["Model", "active [B]", "text total [B]", "GB/step", "TPOT$_{min}$ [ms]",
                             "tok/s$_{max}$", "KV@32k [GiB]", "TTFT$_{min}$@4k [ms]", "draft/target bytes"],
              rows, "lrrrrrrrr")
    if lat.empty:
        return
    # measured single-stream summary
    rows = []
    for m in models_in(lat):
        s = lat[(lat.model == m) & (lat.variant == "base")]
        if s.empty:
            continue
        c = cost(m)
        short = s[(s.kind == "synthetic") & (s.prompt_tokens < 2000)]
        long_ = s[(s.kind == "babilong") & (s.length_config == "32k")]
        tp = short.tpot_s.median()
        ctx = (short.prompt_tokens + 64).median()
        bwu = c.step_bytes(ctx) / tp / BW * 100
        fl = s[s.prompt_tokens > 8000]
        mfu = np.median([c.prefill_flops(p) / t / PEAK * 100 for p, t in zip(fl.prompt_tokens, fl.ttft_s)]) \
            if not fl.empty else np.nan
        sp = mtp_pairs(lat)
        sp = sp[(sp.model == m) & (sp.variant == "mtp4") & (sp.kind != "babilong")]
        cm = conc[(conc.model == m) & (conc.variant == "base")] if not conc.empty else conc
        peak_tp = cm.output_tok_s.max() if not cm.empty else np.nan
        rows.append([LABEL[m], f"{short.ttft_s.median()*1e3:.0f}",
                     f"{long_.ttft_s.median()*1e3:.0f}" if not long_.empty else "--",
                     f"{mfu:.0f}\\%", f"{tp*1e3:.2f}", f"{c.tpot_floor(ctx)*1e3:.2f}", f"{bwu:.0f}\\%",
                     f"{sp.speedup_tpot.median():.2f}$\\times$" if not sp.empty else "--",
                     f"{sp.al.mean():.2f}" if not sp.empty else "--",
                     f"{peak_tp:.0f}" if peak_tp == peak_tp else "--"])
    tex_table("measured.tex", ["Model", "TTFT 1k", "TTFT 32k", "prefill MFU", "TPOT", "TPOT$_{min}$",
                               "HBM util.", "MTP speedup", "AL", "max out tok/s"], rows, "lrrrrrrrrr")
    # speculative-method comparison (per workload kind)
    j = mtp_pairs(lat)
    rows = []
    kinds = [("synthetic", "Synthetic"), ("swe_bench_lite", "SWE-bench"), ("babilong", "Babilong*")]
    for m in models_in(j):
        for v in sorted(j[j.model == m].variant.unique(), key=lambda x: (x.startswith("ngram"), int(x[-1]))):
            s_ = j[(j.model == m) & (j.variant == v)].groupby("kind")
            cells = []
            for k, _ in kinds:
                if k in s_.groups:
                    cells += [f"{s_.al.mean()[k]:.2f}", f"{s_.speedup_tpot.median()[k]:.2f}$\\times$"]
                else:
                    cells += ["--", "--"]
            meth = ("n-gram" if v.startswith("ngram") else "MTP") + f" $\\gamma$={v[-1]}"
            rows.append([LABEL[m], meth] + cells)
    tex_table("spec_compare.tex", ["Model", "method"] + sum([[f"{n} AL", f"{n} speedup"] for _, n in kinds], []),
              rows, "llrrrrrr")
    # KV capacity reported by vLLM: parsed from the server logs (E4) into data/kv_capacity.json,
    # which is what the table is built from, so the logs themselves need not be kept.
    import re
    kv_file = DATA / "kv_capacity.json"
    kv = json.loads(kv_file.read_text()) if kv_file.exists() else {}
    for tag in [t for t in ORDER if t in meta]:
        log = (ROOT / "logs" / f"server_{tag}.log")
        if not log.exists():
            continue
        txt = log.read_text(errors="ignore")
        a = re.search(r"Available KV cache memory: ([\d.]+) GiB", txt)
        n = re.search(r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", txt)
        if a and n:
            kv[tag] = {"kv_pool_gib": float(a.group(1)), "kv_cache_tokens": int(n.group(1).replace(",", "")),
                       "max_model_len": int(n.group(2).replace(",", "")), "max_concurrency": float(n.group(3))}
    kv_file.write_text(json.dumps(kv, indent=1))
    rows = []
    for tag in [t for t in ORDER if t in kv]:
        gib, L, conc_v = kv[tag]["kv_pool_gib"], kv[tag]["max_model_len"], kv[tag]["max_concurrency"]
        c = cost(tag)
        ideal = th.kv_bytes_stored(c.layers, L)
        naive = sum(l["kv_bytes_per_tok"] * L for l in c.layers if l["owns_kv"])
        rows.append([LABEL[tag], f"{gib:.1f}", f"{L//1024}k",
                     f"{ideal/2**30:.2f}", f"{conc_v:.1f}", f"{gib*2**30/ideal:.0f}",
                     f"{gib*2**30/naive:.1f}"])
    tex_table("kv_capacity.tex", ["Model", "KV pool [GiB]", "req. len", "ideal KV/req [GiB]",
                                  "max conc. (vLLM)", "max conc. (ideal)", "max conc. (no window)"],
              rows, "lrrrrrr")


def fig_attn_kernel():
    # Claim: the FA4 decode kernel is pinned at one SM's bandwidth because split-KV does not split;
    # with the split fixed it scales with the number of CTAs and approaches FA3.
    f0, f1 = DATA / "kernel" / "attn_bench.json", DATA / "kernel" / "attn_bench_fix.json"
    if not (f0.exists() and f1.exists()):
        return
    orig, fix = pd.DataFrame(json.loads(f0.read_text())), pd.DataFrame(json.loads(f1.read_text()))
    sel = lambda d, k, hd: d[(d.kernel == k) & (d.head_dim == hd) & (d.page == 16) & (d.num_splits == 0)
                             & (d.kv_heads == 1)].sort_values("seqlen")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    ax = axes[0]
    for d, lab, kw in ((sel(orig, "fa4", 512), "FA4, head dim 512 (vLLM 0.32)", dict(marker="o", ls="-")),
                       (sel(fix, "fa4", 512), "FA4, head dim 512, split fixed", dict(marker="o", ls="--", mfc="white")),
                       (sel(orig, "fa3", 256), "FA3, head dim 256", dict(marker="s", ls="-.", color="#666666"))):
        ax.plot(d.seqlen, d.gbs, label=lab, **{"color": INK, **kw})
    if MB:
        ax.axhline(MB["read_bw_tbs"] * 1e3, color=MUTED, ls=":", lw=1)
        ax.text(4000, MB["read_bw_tbs"] * 1e3 * 0.62, "measured HBM read BW", fontsize=7, color=MUTED)
    ax.set(xscale="log", yscale="log", xlabel="KV length [tokens]", ylabel="effective KV read BW [GB/s]",
           ylim=(10, 40000), title="Decode attention, 1 query, 1 KV head")
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    for d, lab, kw in ((orig, "vLLM 0.32", dict(marker="o", ls="-")),
                       (fix, "split fixed", dict(marker="o", ls="--", mfc="white"))):
        s = d[(d.kernel == "fa4") & (d.head_dim == 512) & (d.page == 16) & (d.seqlen == 30015)
              & (d.num_splits > 0)].sort_values("num_splits")
        ax.plot(s.num_splits, s.us, label=lab, color=INK, **kw)
    s1 = fix[(fix.kernel == "fa4") & (fix.head_dim == 512) & (fix.page == 16) & (fix.seqlen == 30015) & (fix.num_splits == 1)].us.iloc[0]
    xs = np.array([1, 8, 32, 128])
    ax.plot(xs, s1 / xs, color=MUTED, ls=":", lw=1, label="1 split / N")
    ax.set(xscale="log", yscale="log", xlabel="num_splits (CTAs)", ylabel="kernel time [µs]",
           title="Head dim 512, 30k tokens")
    ax.set_xticks(xs)
    ax.legend(fontsize=7)
    save(fig, "attn_kernel")


def fig_gemv():
    # Claim: an M=1 GEMM needs tens of MB of weights to approach HBM bandwidth; the matrices of
    # small models sit on the ramp where a fixed per-kernel time dominates.
    f = DATA / "kernel" / "gemv_bench.json"
    if not f.exists():
        return
    g = json.loads(f.read_text())
    d = pd.DataFrame(g["gemv"])
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for K, kw in ((1536, dict(marker="o", ls="-")), (5376, dict(marker="s", ls="--", mfc="white"))):
        s = d[d.K == K].sort_values("mb")
        ax.plot(s.mb, s.gbs / 1e3, color=INK, label=f"K = {K}", **kw)
    # t = t0 + W / BW fitted on K = 1536 (the hidden size of E2B)
    s = d[d.K == 1536]
    bw_inv, t0 = np.polyfit(s.mb, s.us, 1)
    w = np.geomspace(0.5, 1000, 100)
    ax.plot(w, w / (t0 + bw_inv * w) / 1e3, color=MUTED, ls="-.", lw=0.9,
            label=f"fit: {t0:.1f} µs + W / {1 / bw_inv:.2f} TB/s")
    if MB:
        ax.axhline(MB["read_bw_tbs"], color=MUTED, ls=":", lw=1)
        ax.text(0.55, MB["read_bw_tbs"] + 0.08, "measured HBM read BW", fontsize=7, color=MUTED)
    ax.set(xscale="log", xlabel="weight bytes per GEMM [MB]", ylabel="achieved BW [TB/s]", ylim=(0, 4),
           title=f"M=1 GEMM in a CUDA graph (empty kernel {g['empty_kernel_us']:.1f} µs)")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "gemv_bw")


def main():
    fig_attn_kernel()
    fig_gemv()
    lat, conc, meta = load()
    print(f"latency rows={len(lat)} concurrency rows={len(conc)} tags={sorted(meta)}")
    fig_kv(meta)
    fig_nsys()
    if not lat.empty:
        fig_prefill_rate(lat)
        fig_prefill_hybrid(lat)
        fig_decode_bw(lat)
        fig_decode_gap()
        fig_decode_long(lat)
        fig_mtp(lat)
        fig_gamma(lat)
        fig_spec_model(lat)
        fig_babilong_acc(lat)
    fig_concurrency(conc)
    tables(lat, conc, meta)
    # machine-readable summary for the write-up
    if not lat.empty:
        j = mtp_pairs(lat)
        summ = j.groupby(["model", "variant", "kind"]).agg(
            speedup=("speedup_tpot", "median"), al=("al", "mean"), n=("al", "size")).round(3)
        print(summ.to_string())
        summ.to_csv(DATA / "mtp_summary.csv")


if __name__ == "__main__":
    main()
