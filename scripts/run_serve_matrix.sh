#!/usr/bin/env bash
# Full online-serving matrix (vllm serve + streaming client). Skips configs whose
# result JSON already exists, so it can be re-run after a crash.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export VLLM_CACHE_ROOT="$ROOT/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/torchinductor"
export TRITON_CACHE_DIR="$ROOT/.cache/triton"
[[ -f data/workloads.json ]] || python scripts/prepare_workloads.py

run() {  # tag, then serve_bench.py args
  local tag="$1"; shift
  if [[ -f "data/serve/${tag}.json" ]] && grep -q '"concurrency"\|"latency"' "data/serve/${tag}.json"; then
    echo "#### skip ${tag} (done)"; return 0
  fi
  echo "#### RUN ${tag} $(date -Is)"
  python scripts/serve_bench.py --tag "$tag" "$@" || echo "#### FAIL ${tag}"
}

G=google/gemma-4
# --- baselines + MTP gamma=4 (latency + concurrency)
for m in E2B E4B 12B 26B-A4B; do
  t=${m//-/_}
  run "${t}"       --model "$G-$m-it"
  run "${t}_mtp4"  --model "$G-$m-it" --spec-method mtp --draft-model "$G-$m-it-assistant" --num-spec 4
done
# 31B dense: BF16 weights ~58 GiB -> tighter context
run 31B      --model "$G-31B-it" --max-model-len 34816 --gpu-mem-util 0.95
run 31B_mtp4 --model "$G-31B-it" --max-model-len 34816 --gpu-mem-util 0.95 \
             --spec-method mtp --draft-model "$G-31B-it-assistant" --num-spec 4

# --- gamma sweep (latency only)
for n in 1 2 8; do
  run "E2B_mtp${n}"     --suites latency --model "$G-E2B-it" --spec-method mtp --draft-model "$G-E2B-it-assistant" --num-spec $n
  run "26B_A4B_mtp${n}" --suites latency --model "$G-26B-A4B-it" --spec-method mtp --draft-model "$G-26B-A4B-it-assistant" --num-spec $n
  run "31B_mtp${n}"     --suites latency --model "$G-31B-it" --max-model-len 34816 --gpu-mem-util 0.95 \
                        --spec-method mtp --draft-model "$G-31B-it-assistant" --num-spec $n
done
# --- model-free speculative baseline (prompt lookup) for comparison
run 31B_ngram4 --suites latency --model "$G-31B-it" --max-model-len 34816 --gpu-mem-util 0.95 \
               --spec-method ngram --num-spec 4

# --- other model families
run Qwen3_30B_A3B --model Qwen/Qwen3-30B-A3B-Instruct-2507
run Qwen3_32B     --model Qwen/Qwen3-32B --max-model-len 40960 --gpu-mem-util 0.95
echo "#### MATRIX DONE $(date -Is)"
