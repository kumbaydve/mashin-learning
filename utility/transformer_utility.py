from parts.single_head_attention import SingleHeadAttention


def get_heads(embedding_d, head_d, seq_len, optimizer_class, optimizer_parameters):
	return [SingleHeadAttention(embedding_d, head_d, seq_len, optimizer_class(*optimizer_parameters)) for _ in range(embedding_d // head_d)]
