# vllm/model_executor/models/gemma4_mtp.py  class Gemma4MTPMaskedEmbedder (excerpt)
def _select_and_score(self, hidden_states, lm_head_weight):
    _, top_k_indices = torch.topk(self.centroids(hidden_states),     # <- score 2048 centroids
                                  k=self.centroid_intermediate_top_k, dim=-1)  # <- keep 32
    clusters = self.token_ordering.view(self.num_centroids, self.vocab_size_per_centroid)
    selected = clusters[top_k_indices]                                # <- 32 x 128 = 4096 tokens
    embeddings = lm_head_weight[selected.reshape(-1)].view(
        num_tokens, self.num_selected, self.hidden_size)              # <- read only these rows
    logits = torch.einsum("td,tsd->ts", hidden_states, embeddings)
