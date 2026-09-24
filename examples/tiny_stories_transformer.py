import json
import torch

from constants import SPECIAL_CHARS, SPACE_CHARS
from oleguses.olegus_transformer import OlegusTransformer
from parts.optimizers import Adam

from utility.transformer_utility import str_to_tokens, tokens_to_ixs, ixs_to_subsequences, ixs_to_next_ixs, \
	generate_from_tokens


dictionary = set()

with open('TinyStories-train.txt', 'r', encoding='utf-8') as file:
	while line := file.readline():
		line = line.strip().lower()

		for token in str_to_tokens(line, SPECIAL_CHARS, SPACE_CHARS, ('<|endoftext|>',)):
			dictionary.add(token)

dictionary = list(dictionary)

with open('tiny_stories_dictionary.json', 'w', encoding='utf-8') as file:
	json.dump(dictionary, file)


text = ''

with open('TinyStories-train.txt', 'r', encoding='utf-8') as file:
	for _ in range(3_000):
		line = file.readline().strip()
		text += line + ' '

text = text.lower().strip()
tokens = str_to_tokens(text, SPECIAL_CHARS, SPACE_CHARS, ('<|endoftext|>',))


embedding_d = 128 + 64
head_d = 32
seq_len = 64
perceptron_d = 512
layer_n = 2

olegus = OlegusTransformer(dictionary, embedding_d, head_d, perceptron_d, seq_len, layer_n, Adam, (0.001, 0.9, 0.999))

ixs = tokens_to_ixs(tokens, olegus)
seqs = ixs_to_subsequences(ixs, seq_len)
nexts = ixs_to_next_ixs(ixs, seq_len)

tests = [
	'once upon a time',
	'once upon a time, there was a girl. she was',
	'once upon a time, there was a boy. he was',
	'one day, a girl named',
	'one day, a boy named',
	'once upon a time, there was a girl. she loved to explore. one day, she was',
	''
]

for epoch in range(1, 2 + 1):
	print('EPOCH', epoch)

	shuffled_seq_ixs = torch.randperm(seqs.shape[0])

	total_loss = 0
	n = 0

	for i in shuffled_seq_ixs:
		if (n + 1) % 1_000 == 0:
			print(f'{n + 1} / {len(seqs)}')
			print('loss', total_loss / 1_000)

			total_loss = 0

		olegus.drop_gradient()
		olegus.forward(seqs[i, :], 1)
		total_loss += olegus.get_loss(nexts[i])
		olegus.backward(nexts[i])
		olegus.descent()

		n += 1

	print('TESTS')

	for text in tests:
		for temperature in [0.5, 1, 2]:
			tokens = str_to_tokens(text, ',', ' ')

			print(temperature, *generate_from_tokens(tokens, seq_len - len(tokens), olegus, temperature))

		print()

olegus.save('olegus_transformer_tiny_stories.json', save_gradients=True)
