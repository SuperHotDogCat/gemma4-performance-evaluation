# Put scripts/fa4fix on PYTHONPATH and set GEMMA4_FA4_FIX=1 to apply the split-KV fix
# (fa4_splitkv_fix.py) in every Python process, including vLLM's engine-core workers.
import os

if os.environ.get("GEMMA4_FA4_FIX") == "1":
    import fa4_splitkv_fix  # noqa: F401
