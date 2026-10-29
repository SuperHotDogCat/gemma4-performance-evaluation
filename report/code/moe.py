# vllm/model_executor/models/gemma4.py  class Gemma4DecoderLayer.forward (excerpt)
# MLP runs unconditionally (same inputs for MoE and non-MoE)
hidden_states = self.pre_feedforward_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)                       # <- dense MLP (2112)
if self.enable_moe_block:
    hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
    hidden_states_2 = self.pre_feedforward_layernorm_2(residual)
    router_logits = self.router(residual)                      # <- logits over 128 experts
    hidden_states_2 = self.experts(hidden_states_2, router_logits)  # <- fused MoE, top-8
    hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
    # Combine MLP and MoE outputs
    hidden_states = hidden_states_1 + hidden_states_2

# class Gemma4Router
def routing_function(self, hidden_states, gating_output, topk, renormalize):
    """Softmax over ALL experts -> top-k -> renormalize."""
    return gemma4_fused_routing_kernel_triton(gating_output, topk, self.per_expert_scale)
