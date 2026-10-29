#!/usr/bin/env bash
# GPU microbench + Nsight Systems decode profiles. Run when the GPU is otherwise idle.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export VLLM_CACHE_ROOT="$ROOT/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/torchinductor"
export TRITON_CACHE_DIR="$ROOT/.cache/triton"
mkdir -p data/nsys logs/nsys

python scripts/microbench.py

prof() {  # tag, profile_decode args
  local tag="$1"; shift
  [[ -f "data/nsys/${tag}.json" ]] && { echo "skip $tag"; return; }
  echo "#### NSYS ${tag}"
  nsys profile -f true -o "logs/nsys/${tag}" -t cuda,nvtx --cuda-graph-trace=node \
       --capture-range=cudaProfilerApi --capture-range-end=stop \
       python scripts/profile_decode.py "$@" 2>&1 | grep -E 'PROFILE_RESULT|Error|error' | tee "logs/nsys/${tag}.txt"
  nsys export -f true --type sqlite -o "logs/nsys/${tag}.sqlite" "logs/nsys/${tag}.nsys-rep" >/dev/null &&
    python scripts/analyze_nsys.py "logs/nsys/${tag}.sqlite" "data/nsys/${tag}.json"
  # attach the wall-clock result
  python - "$tag" <<'PY'
import json, sys, re
tag = sys.argv[1]
txt = open(f"logs/nsys/{tag}.txt").read()
m = re.search(r"PROFILE_RESULT (\{.*\})", txt)
p = f"data/nsys/{tag}.json"
d = json.load(open(p)); d["run"] = json.loads(m.group(1)) if m else None
json.dump(d, open(p, "w"), indent=1)
PY
}

G=google/gemma-4
for m in E2B 12B 26B-A4B 31B; do
  t=${m//-/_}
  prof "${t}_b1"  --model "$G-$m-it" --batch 1
  prof "${t}_b32" --model "$G-$m-it" --batch 32
  prof "${t}_mtp4_b1" --model "$G-$m-it" --batch 1 --draft-model "$G-$m-it-assistant"
done
prof E2B_b1_ctx32k --model "$G-E2B-it" --batch 1 --context 30000 --max-model-len 34000
prof 31B_b1_ctx28k --model "$G-31B-it" --batch 1 --context 26000 --max-model-len 28800 --gpu-mem-util 0.95
prof Qwen3_30B_A3B_b1 --model Qwen/Qwen3-30B-A3B-Instruct-2507 --batch 1
echo "#### PROFILES DONE"
