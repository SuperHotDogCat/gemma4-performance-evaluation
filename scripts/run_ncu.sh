#!/usr/bin/env bash
# Nsight Compute on the decode attention kernel of Gemma 4 E2B at ~30k context.
# Global (head_dim 512) and sliding (256) layers share the same FA4 kernel name, so we
# profile one full decode step (35 consecutive launches) and split them by duration.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export VLLM_CACHE_ROOT="$ROOT/.cache/vllm" TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/torchinductor" TRITON_CACHE_DIR="$ROOT/.cache/triton"
mkdir -p logs/ncu data/ncu
# skip warm-up/prefill launches: profiler range only covers the measured generate();
# within it skip the prefill + first decode steps, then capture 35 attention launches.
sudo -E env PATH="$PATH" HOME=/tmp/ncu_home HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" /opt/nvidia/nsight-compute/2025.3.1/ncu --profile-from-start off -f -o logs/ncu/E2B_attn_ctx32k \
    --kernel-name regex:FlashAttentionForwardSm90 --launch-skip 700 --launch-count 35 \
    --replay-mode application \
    --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size,launch__block_size,launch__registers_per_thread,sm__cycles_active.avg,smsp__inst_executed.sum,lts__t_bytes.sum \
    python scripts/profile_decode.py --model google/gemma-4-E2B-it --batch 1 \
      --context 30000 --max-model-len 34000 --out-tokens 64 --enforce-eager 2>&1 | grep -vE '^\s*$' | tail -5
# the report was written as root: take ownership before reading it back
sudo chown -R "$(id -u):$(id -g)" logs/ncu .cache
/opt/nvidia/nsight-compute/2025.3.1/ncu -i logs/ncu/E2B_attn_ctx32k.ncu-rep --csv --page raw > logs/ncu/E2B_attn_ctx32k.csv \
  || { echo "#### NCU export failed"; exit 1; }
mv logs/ncu/E2B_attn_ctx32k.csv data/ncu/E2B_attn_ctx32k.csv
echo "#### NCU DONE"
