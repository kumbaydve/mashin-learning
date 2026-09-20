import math

import torch

from parts.activations import Softmax
from parts.losses import CrossEntropy
from parts.sequence import Sequence
from parts.transformer_layer import TransformerLayer
from universal import Model


class OlegusTransformer(Model):
	def __init__(self, dictionary, embedding_d, head_d, perceptron_d, seq_len, layer_n, optimizer_class, optimizer_parameters):
		self.dictionary_size = len(dictionary)
		self.embedding_d = embedding_d
		self.head_d = head_d
		self.perceptron_d = perceptron_d
		self.seq_len = seq_len
		self.layer_n = layer_n

		self.token_to_ix = dict()

		for ix in range(self.dictionary_size):
			self.token_to_ix[dictionary[ix]] = ix

		self.embedding_matrix = torch.randn(self.dictionary_size, embedding_d)

		self.layer_sequence = Sequence(
			*[TransformerLayer(embedding_d, head_d, perceptron_d, seq_len, optimizer_class, optimizer_parameters) for _ in range(layer_n)]
		)

		self.w_out = torch.randn((embedding_d, self.dictionary_size)) * math.sqrt(2 / (embedding_d + self.dictionary_size))
		self.b_out = torch.zeros(self.dictionary_size)

		self.activation = Softmax()
		self.loss_function = CrossEntropy()
		self.optimizer = optimizer_class(*optimizer_parameters)

		self.positional_encoding = torch.empty((self.seq_len, self.embedding_d))

		for pos in range(self.seq_len):
			for i in range(0, self.embedding_d, 2):
				self.positional_encoding[pos, i] = torch.sin(torch.tensor(pos / (10000 ** (i / self.embedding_d))))

				if i + 1 < self.embedding_d:
					self.positional_encoding[pos, i + 1] = torch.cos(torch.tensor(pos / (10000 ** (i / self.embedding_d))))

		self.optimizer.connect(self, 'w_out', 'b_out', 'embedding_matrix')

	def forward(self, ixs, temperature):
		self.ixs = ixs
		self.x = self.embedding_matrix[ixs, :]
		self.layers_y = self.layer_sequence.forward(self.x + self.positional_encoding)

		logits = self.layers_y @ self.w_out + self.b_out
		self.y = self.activation.forward(logits / temperature)

		return self.y

	def store_expected(self, next_ix):
		self.expected_ixs = torch.cat((self.ixs[1:], next_ix))
		self.expected = torch.zeros_like(self.y)
		self.expected[torch.arange(self.seq_len), self.expected_ixs] = 1

	def get_loss(self):
		return self.loss_function.forward(self.y, self.expected)

	def backward(self):
		d_in = self.y - self.expected

		self.optimizer.backward(
			b_out = torch.sum(d_in, dim=0),
			w_out = self.layers_y.T @ d_in
		)

		d_in = self.layer_sequence.backward(d_in @ self.w_out.T)

		self.optimizer.backward(
			embedding_matrix = ((self.ixs, slice(self.embedding_d)), d_in)
		)

	def drop_gradient(self):
		self.optimizer.drop_gradient()
		self.layer_sequence.drop_gradient()

	def descent(self):
		self.optimizer.descent()
		self.layer_sequence.descent()
