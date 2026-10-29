# vllm/model_executor/models/gemma4.py  class Gemma4Attention (excerpt)
first_kv_shared_layer_idx = config.num_hidden_layers - num_kv_shared_layers
self.is_kv_shared_layer = (
    num_kv_shared_layers > 0 and layer_idx >= first_kv_shared_layer_idx
)
if self.is_kv_shared_layer:
    # K/V come from the target layer's cache, so like HF this layer
    # only has q_proj (and no k_norm/v_norm).
    self.q_proj = ColumnParallelLinear(hidden_size, self.total_num_heads * self.head_dim, ...)
...
# Find the last non-shared layer of the same attention type
prev_layers = config.layer_types[:first_kv_shared_layer_idx]
current_layer_type = config.layer_types[layer_idx]
kv_shared_layer_index = (
    len(prev_layers) - 1 - prev_layers[::-1].index(current_layer_type)
)                                               # <- E2B: local -> layer 14, global -> layer 15

def forward(self, positions, hidden_states, **kwargs):
    if self.is_kv_shared_layer:
        # Shared KV: only Q is projected; K/V come from the target layer.
        q, _ = self.q_proj(hidden_states)
        ...
        attn_output = self.attn(q, None, None)  # <- nothing is written to the KV cache
