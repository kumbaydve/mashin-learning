from parts.single_head_attention import SingleHeadAttention


def get_heads(embedding_d, head_d, seq_len, head_optimizer_class, head_optimizer_parameters):
	[SingleHeadAttention(embedding_d, head_d, seq_len, head_optimizer_class(*head_optimizer_parameters)) for _ in range(embedding_d // head_d)]
