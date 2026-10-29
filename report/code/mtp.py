# vllm/model_executor/models/gemma4_mtp.py  class Gemma4MultiTokenPredictor (excerpt)
def forward(self, input_ids, positions, hidden_states, ...):
    """Returns (draft_hidden_states, backbone_hidden_states).
    backbone_hidden_states: backbone-dim, stored in the proposer's
        hidden-state buffer and fed back as input to the next step.
    """
    inputs_embeds = self.embed_input_ids(input_ids)
    combined = torch.cat([inputs_embeds, hidden_states], dim=-1)   # <- (1) embedding + hidden
    hidden_states, _ = self.pre_projection(combined)               # <- 2 x backbone -> draft dim
    for layer in self.layers:                                      # <- 4 Q-only layers
        hidden_states, residual = layer(positions=positions, hidden_states=hidden_states, ...)
    draft_hidden_states = self.norm(hidden_states)                 # <- -> LM head (draft dim)
    backbone_hidden_states, _ = self.post_projection(draft_hidden_states)  # <- (2) fed back
    return draft_hidden_states, backbone_hidden_states

# vllm/v1/spec_decode/gemma4.py  (excerpt)
# All draft steps predict from the same position (the last
# target-model position), so positions and seq_lens must not
# advance between steps.
self.constant_draft_positions = True
...
# Each draft decoder layer is mapped to the last non-KV-shared
# target layer of the same attention type (sliding or full).
target_idx = candidates[-1]                                        # <- (3)
attn.kv_sharing_target_layer_name = f"{target_prefix}.{target_idx}.self_attn.attn"
