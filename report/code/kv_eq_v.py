# vllm/model_executor/models/gemma4.py  class Gemma4Attention (excerpt)
sliding_window = config.sliding_window if self.is_sliding else None
self.attn = Attention(..., per_layer_sliding_window=sliding_window, ...)  # <- local: window, global: None

def forward(self, positions, hidden_states, **kwargs):
    ...
    # For k_eq_v, K weights are loaded into both K and V slots of
    # qkv_proj, so V == K automatically.
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    ...
    k = self.k_norm(k)                          # <- K: RMSNorm with weight
    q, k = self.rotary_emb(positions, q, k)     # <- K: RoPE
    v = self.v_norm(v)                          # <- V: RMSNorm without weight, no RoPE
    attn_output = self.attn(q, k, v)            # <- K and V are cached separately
