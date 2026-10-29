#!/usr/bin/env bash
# Re-measurement of the doubtful points R1-R5 (backend, FP8 KV, MTP repeatability, Qwen3 long context, microbench).
# Results go to data/remeasure/ and logs/remeasure/ so that data/serve (read by analyze.py)
# is untouched. Skips steps whose output already exists, so it can be re-run after a crash.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export VLLM_CACHE_ROOT="$ROOT/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/torchinductor"
export TRITON_CACHE_DIR="$ROOT/.cache/triton"
[[ -f data/workloads.json ]] || python scripts/prepare_workloads.py
OUT=data/remeasure
LOG=logs/remeasure
mkdir -p "$OUT/serve" "$OUT/nsys" "$LOG/nsys"
G=google/gemma-4

step() { echo "#### $(date -Is) $*"; }

serve() {  # tag, then serve_bench.py args
  local tag="$1"; shift
  if [[ -f "$OUT/serve/${tag}.json" ]] && grep -q '"latency"\|"concurrency"' "$OUT/serve/${tag}.json"; then
    step "skip ${tag} (done)"; return 0
  fi
  step "RUN ${tag}"
  python scripts/serve_bench.py --tag "$tag" --out-dir "$OUT/serve" --log-dir "$LOG" "$@" \
    || step "FAIL ${tag}"
}

prof() {  # tag, profile_decode.py args
  local tag="$1"; shift
  [[ -f "$OUT/nsys/${tag}.json" ]] && { step "skip nsys ${tag}"; return; }
  step "NSYS ${tag}"
  nsys profile -f true -o "$LOG/nsys/${tag}" -t cuda,nvtx --cuda-graph-trace=node \
       --capture-range=cudaProfilerApi --capture-range-end=stop \
       python scripts/profile_decode.py "$@" > "$LOG/nsys/${tag}.txt" 2>&1
  nsys export -f true --type sqlite -o "$LOG/nsys/${tag}.sqlite" "$LOG/nsys/${tag}.nsys-rep" >/dev/null &&
    python scripts/analyze_nsys.py "$LOG/nsys/${tag}.sqlite" "$OUT/nsys/${tag}.json" || step "FAIL nsys ${tag}"
}

# R5: GEMM peak / HBM BW stability (report text once said 525 TFLOPS, data says 492)
for i in 1 2 3; do
  [[ -f "$OUT/microbench_${i}.json" ]] || { step "MICROBENCH ${i}"; python scripts/microbench.py --out "$OUT/microbench_${i}.json" > "$LOG/microbench_${i}.txt" 2>&1; }
done

# R1: 26B-A4B MTP speedup (1.35x here vs 1.49-1.60x externally) -- repeatability, gamma 4 and 8
serve 26B_A4B      --suites latency --model "$G-26B-A4B-it"
serve 26B_A4B_mtp4 --suites latency --model "$G-26B-A4B-it" --spec-method mtp --draft-model "$G-26B-A4B-it-assistant" --num-spec 4
serve 26B_A4B_mtp8 --suites latency --model "$G-26B-A4B-it" --spec-method mtp --draft-model "$G-26B-A4B-it-assistant" --num-spec 8
prof 26B_A4B_mtp4_b1 --model "$G-26B-A4B-it" --batch 1 --draft-model "$G-26B-A4B-it-assistant"

# R2: is the long-context decode slowdown the FA4 kernel? same model, other backends
serve E2B_triton --suites latency --model "$G-E2B-it" --extra-server-args --attention-backend TRITON_ATTN
serve E2B_fp8kv  --suites latency --model "$G-E2B-it" --extra-server-args --kv-cache-dtype fp8

# R3: Qwen3-32B long-context attention, measured instead of estimated from TPOT (Table 12)
prof Qwen3_32B_b1          --model Qwen/Qwen3-32B --batch 1 --gpu-mem-util 0.95
prof Qwen3_32B_b1_ctx28k   --model Qwen/Qwen3-32B --batch 1 --context 26000 --max-model-len 28800 --gpu-mem-util 0.95

# R4: 31B -- backend effect at long context, and FP8 KV for the KV-capacity recommendation
serve 31B_triton --suites latency --model "$G-31B-it" --max-model-len 34816 --gpu-mem-util 0.95 \
                 --extra-server-args --attention-backend TRITON_ATTN
serve 31B_fp8kv  --model "$G-31B-it" --max-model-len 34816 --gpu-mem-util 0.95 \
                 --extra-server-args --kv-cache-dtype fp8
step "REMEASURE DONE"
