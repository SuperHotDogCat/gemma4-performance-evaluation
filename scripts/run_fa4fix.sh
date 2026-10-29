#!/usr/bin/env bash
# E9: kernel-level analysis of the long-context decode attention, and the effect of the
# split-KV fix (scripts/fa4fix) on end-to-end serving. Results go to data/kernel/ and
# data/fa4fix/ (not read by analyze.py's main figures). Skips steps whose output exists.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export VLLM_CACHE_ROOT="$ROOT/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/torchinductor"
export TRITON_CACHE_DIR="$ROOT/.cache/triton"
[[ -f data/workloads.json ]] || python scripts/prepare_workloads.py
OUT=data/fa4fix
LOG=logs/fa4fix
mkdir -p "$OUT/serve" "$OUT/nsys" "$LOG/nsys" data/kernel logs/kernel
G=google/gemma-4
FIXENV=(env PYTHONPATH="$ROOT/scripts/fa4fix${PYTHONPATH:+:$PYTHONPATH}" GEMMA4_FA4_FIX=1)

step() { echo "#### $(date -Is) $*"; }

# kernel microbenchmarks (vLLM 0.32 as is / with the fix) and decode GEMV sizes
[[ -f data/kernel/probe_E2B.json ]] || python scripts/attn_probe.py --model "$G-E2B-it" --out data/kernel/probe_E2B.json > logs/kernel/probe_E2B.log 2>&1
[[ -f data/kernel/attn_bench.json ]] || { step "ATTN BENCH"; python scripts/attn_bench.py --out data/kernel/attn_bench.json > logs/kernel/attn_bench.log 2>&1; }
[[ -f data/kernel/attn_bench_fix.json ]] || { step "ATTN BENCH FIX"; python scripts/attn_bench.py --fix --out data/kernel/attn_bench_fix.json > logs/kernel/attn_bench_fix.log 2>&1; }
[[ -f data/kernel/gemv_bench.json ]] || { step "GEMV BENCH"; python scripts/gemv_bench.py --out data/kernel/gemv_bench.json > logs/kernel/gemv_bench.log 2>&1; }

serve() {  # tag, then serve_bench.py args (run with the fix)
  local tag="$1"; shift
  if [[ -f "$OUT/serve/${tag}.json" ]] && grep -q '"latency"\|"concurrency"' "$OUT/serve/${tag}.json"; then
    step "skip ${tag} (done)"; return 0
  fi
  step "RUN ${tag}"
  "${FIXENV[@]}" python scripts/serve_bench.py --tag "$tag" --out-dir "$OUT/serve" --log-dir "$LOG" "$@" \
    || step "FAIL ${tag}"
}

prof() {  # tag, profile_decode.py args (run with the fix)
  local tag="$1"; shift
  [[ -f "$OUT/nsys/${tag}.json" ]] && { step "skip nsys ${tag}"; return; }
  step "NSYS ${tag}"
  "${FIXENV[@]}" nsys profile -f true -o "$LOG/nsys/${tag}" -t cuda,nvtx --cuda-graph-trace=node \
       --capture-range=cudaProfilerApi --capture-range-end=stop \
       python scripts/profile_decode.py "$@" > "$LOG/nsys/${tag}.txt" 2>&1
  nsys export -f true --type sqlite -o "$LOG/nsys/${tag}.sqlite" "$LOG/nsys/${tag}.nsys-rep" >/dev/null &&
    python scripts/analyze_nsys.py "$LOG/nsys/${tag}.sqlite" "$OUT/nsys/${tag}.json" || step "FAIL nsys ${tag}"
}

# same options as the original E1/E6 runs, only the fix differs
prof E2B_b1_ctx32k --model "$G-E2B-it" --batch 1 --context 30000 --max-model-len 34000 --out-tokens 64
serve E2B     --suites latency --model "$G-E2B-it"
serve 31B     --model "$G-31B-it" --max-model-len 34816 --gpu-mem-util 0.95
serve E4B     --suites latency --model "$G-E4B-it"
serve 12B     --suites latency --model "$G-12B-it"
serve 26B_A4B --suites latency --model "$G-26B-A4B-it"
prof 31B_b1_ctx28k --model "$G-31B-it" --batch 1 --context 26000 --max-model-len 28800 --gpu-mem-util 0.95 --out-tokens 64
step "FA4FIX DONE"
