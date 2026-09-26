import torch
from torch.utils.benchmark.utils.fuzzer import dtype_size

from constants import INT, ERROR_TOKEN, ERROR_TOKEN_IX
from parts.single_head_attention import SingleHeadAttention


def get_heads(embedding_d, head_d, seq_len, optimizer_class, optimizer_parameters):
	return [SingleHeadAttention(embedding_d, head_d, seq_len, optimizer_class(*optimizer_parameters)) for _ in range(embedding_d // head_d)]


buffer = ''

def str_to_tokens(text, special_chars, space_chars, special_tokens=tuple()):
	parts = [text]

	for special_token in special_tokens:
		new_parts = []

		for part in parts:
			splited_by_token = list(map(str.strip, part.split(special_token)))

			for splited_by_token_part in splited_by_token[:-1]:
				if splited_by_token_part:
					new_parts.append(splited_by_token_part)

				new_parts.append(special_token)

			if splited_by_token[-1]:
				new_parts.append(splited_by_token[-1])

		parts = new_parts

	global buffer
	res = []
	buffer = ''


	def flush():
		global buffer

		if buffer:
			res.append(buffer)

		buffer = ''


	for part in parts:
		if part not in special_tokens:
			for char in part:
				if char in special_chars:
					flush()
					res.append(char)
				elif char in space_chars:
					flush()
				else:
					buffer += char

			flush()
		else:
			res.append(part)

	return res


def tokens_to_dictionary(tokens):
	res = list(set(tokens))

	return res[:ERROR_TOKEN_IX] + [ERROR_TOKEN] + res[ERROR_TOKEN_IX:]


def tokens_to_ixs(tokens, olegus):
	res = torch.empty(len(tokens), dtype=INT)

	for i, token in enumerate(tokens):
		if token in olegus.token_to_ix:
			res[i] = olegus.token_to_ix[token]
		else:
			res[i] = ERROR_TOKEN_IX

	return res


def ixs_to_subsequences(ixs, length):
	res = torch.empty((ixs.shape[0] - length + 1, length), dtype=INT)

	for i in range(ixs.shape[0] - length + 1):
		res[i, :] = ixs[i:i+length]

	return res


def ixs_to_next_ixs(ixs, length):
	res = torch.empty(ixs.shape[0] - length + 1, dtype=INT)

	for i in range(ixs.shape[0] - length):
		res[i] = ixs[i + length]

	res[-1] = ixs[0]

	return res


def y_to_next_token(y, dictionary, n, min_p, after_word_at_ix=-1):
	ixs = torch.argsort(y[after_word_at_ix, :])[-n:]
	ixs = ixs[y[after_word_at_ix, ixs] >= min_p]

	if ixs.numel() == 0:
		return ERROR_TOKEN

	ix_of_ixs = torch.multinomial(y[after_word_at_ix, ixs], num_samples=1)
	return dictionary[ixs[ix_of_ixs]]


def y_to_next_possible_tokens(y, dictionary, n, min_p, after_word_at_ix=-1):
	ixs = torch.argsort(y[after_word_at_ix, :])[-n:]
	ixs = ixs[y[after_word_at_ix, ixs] >= min_p]

	return [dictionary[ix] for ix in torch.flip(ixs, dims=[0])]


def generate_from_tokens(tokens, num_of_words, olegus, temperature=1):
	for i in range(num_of_words):
		ixs = torch.cat((tokens_to_ixs(tokens, olegus), torch.tensor([0] * (olegus.seq_len - len(tokens)))))
		y = olegus.forward(ixs, temperature)
		next_token = y_to_next_token(y, olegus.dictionary, 5, 0.05, len(tokens) - 1)
		tokens.append(next_token)

	return tokens
