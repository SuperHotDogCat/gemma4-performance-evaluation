# vllm/model_executor/models/config.py  class Gemma4Config (excerpt)
def verify_and_update_config(vllm_config):
    """Configure attention for heterogeneous head dimensions.

    Gemma4 uses different head dimensions for sliding window vs full attention
    layers. The default FA3 on Hopper cannot handle head_dim > 256.
    ...
    force FA4 for all layers to avoid the mixed FA3+FA4 penalty.
    """
    ...
    if is_fa_version_supported(4) and max_head_dim <= 512:      # <- global layers: 512
        ...
        if use_per_layer_fa:        # <- only on SM90 *with an FP8 KV cache*
            ...                     #    (FA3 for 256-dim layers, FA4 for 512-dim layers)
        else:
            vllm_config.attention_config.flash_attn_version = 4  # <- BF16 KV: FA4 for every layer
