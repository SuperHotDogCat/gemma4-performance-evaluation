# vllm/model_executor/models/gemma4.py  (excerpt)
# (a) once per token: build e_1..e_L
def get_per_layer_inputs(self, input_ids):
    per_layer_embeds = self.embed_tokens_per_layer(per_layer_inputs_tokens)  # <- row lookup only
    per_layer_embeds = per_layer_embeds * self.embed_scale_per_layer
    return per_layer_embeds.reshape(*input_ids.shape, self.config.num_hidden_layers,
                                    self.hidden_size_per_layer_input)

def project_per_layer_inputs(self, inputs_embeds, per_layer_inputs):
    per_layer_projection = self.per_layer_model_projection(inputs_embeds)
    per_layer_projection = per_layer_projection * self.per_layer_projection_scale
    per_layer_projection = per_layer_projection.reshape(...)
    per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
    return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale  # <- 1/sqrt(2)

# (b) class Gemma4DecoderLayer.forward: after attention and MLP
if per_layer_input is not None and self.per_layer_input_gate is not None:
    gate = self.per_layer_input_gate(hidden_states)           # <- 1536 -> 256
    gate = torch.nn.functional.gelu(gate, approximate="tanh")
    gated_per_layer = gate * per_layer_input                  # <- elementwise with e_l
    per_layer_contribution = self.per_layer_projection(gated_per_layer)  # <- 256 -> 1536
    per_layer_contribution = self.post_per_layer_input_norm(per_layer_contribution)
    hidden_states = hidden_states + per_layer_contribution
