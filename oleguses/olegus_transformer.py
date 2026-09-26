# EINSUM


import math

import torch

from parts.activations import Softmax
from parts.losses import CrossEntropy
from parts.optimizers import Optimizer
from parts.sequence import Sequence
from parts.transformer_layer import TransformerLayer
from universal import Model, Savable


class OlegusTransformer(Model, Savable):
	def __init__(self, dictionary, embedding_d, head_d, perceptron_d, seq_len, layer_n, optimizer_class, optimizer_parameters, from_obj=False):
		self.dictionary = dictionary
		self.dictionary_size = len(dictionary)
		self.embedding_d = embedding_d
		self.head_d = head_d
		self.perceptron_d = perceptron_d
		self.seq_len = seq_len
		self.layer_n = layer_n

		self.token_to_ix = dict()

		for ix in range(self.dictionary_size):
			self.token_to_ix[dictionary[ix]] = ix

		if not from_obj:
			self.embedding_matrix = torch.randn(self.dictionary_size, embedding_d)

			self.layer_sequence = Sequence(
				*[TransformerLayer(embedding_d, head_d, perceptron_d, seq_len, optimizer_class, optimizer_parameters) for _ in range(layer_n)]
			)

			self.w_out = torch.randn((embedding_d, self.dictionary_size)) * math.sqrt(2 / (embedding_d + self.dictionary_size))
			self.b_out = torch.zeros(self.dictionary_size)

			self.optimizer = optimizer_class(*optimizer_parameters)
			self.optimizer.connect(self, 'w_out', 'b_out', 'embedding_matrix')

		self.activation = Softmax()
		self.loss_function = CrossEntropy()

		self.positional_encoding = torch.empty((self.seq_len, self.embedding_d))

		for pos in range(self.seq_len):
			for i in range(0, self.embedding_d, 2):
				self.positional_encoding[pos, i] = torch.sin(torch.tensor(pos / (10000 ** (i / self.embedding_d))))

				if i + 1 < self.embedding_d:
					self.positional_encoding[pos, i + 1] = torch.cos(torch.tensor(pos / (10000 ** (i / self.embedding_d))))

	def forward(self, ixs, temperature):
		self.ixs = ixs
		self.x = self.embedding_matrix[ixs, :]
		self.layers_y = self.layer_sequence.forward(self.x + self.positional_encoding)

		logits = self.layers_y @ self.w_out + self.b_out
		self.y = self.activation.forward(logits / temperature)

		return self.y

	def get_loss(self, next_ix):
		#print(self.y.shape)
		#print(self.ixs.shape)
		#print(torch.cat((self.ixs[:, 1:], next_ix.unsqueeze(1)), dim=-1).shape)
		return torch.mean(-torch.log(self.y[:, :, torch.cat((self.ixs[:, 1:], next_ix.unsqueeze(1)), dim=-1)]))

	def backward(self, next_ix):
		d_in = self.y.detach().clone() # bsd
		d_in[:, :, torch.cat((self.ixs[:, 1:], next_ix.unsqueeze(1)), dim=-1)] -= 1

		self.optimizer.backward(
			b_out = torch.einsum('...sd->d', d_in), # bsd -> d
			w_out = torch.einsum('...es,...sd->ed', self.layers_y.mT, d_in) # bes, bsd -> ed
		)

		d_in = self.layer_sequence.backward(d_in @ self.w_out.mT) # bsd, de -> bse

		for i in range(self.ixs.shape[0]):
			self.optimizer.backward(
				embedding_matrix = ((self.ixs[i], slice(self.embedding_d)), d_in[i])
			)

	def drop_gradient(self):
		self.optimizer.drop_gradient()
		self.layer_sequence.drop_gradient()

	def descent(self):
		self.optimizer.descent()
		self.layer_sequence.descent()

	def to_obj(self, save_gradients=False):
		return {
			'name': self.__class__.__name__,
			'dictionary': self.dictionary,
			'embedding_d': self.embedding_d,
			'head_d': self.head_d,
			'perceptron_d': self.perceptron_d,
			'seq_len': self.seq_len,
			'layer_n': self.layer_n,
			'embedding_matrix': self.embedding_matrix.tolist(),
			'layer_sequence': self.layer_sequence.to_obj(save_gradients=save_gradients),
			'w_out': self.w_out.tolist(),
			'b_out': self.b_out.tolist(),
			'optimizer': self.optimizer.to_obj(save_gradients=save_gradients)
		}

	@staticmethod
	def from_obj(obj, load_gradients=True):
		res = OlegusTransformer(obj['dictionary'], obj['embedding_d'], obj['head_d'], obj['perceptron_d'], obj['seq_len'], obj['layer_n'], None, None, from_obj=True)

		res.embedding_matrix = torch.tensor(obj['embedding_matrix'])

		res.layer_sequence = Sequence.from_obj(obj['layer_sequence'], load_gradients=load_gradients)

		res.w_out = torch.tensor(obj['w_out'])
		res.b_out = torch.tensor(obj['b_out'])

		res.optimizer = Optimizer.from_obj(obj['optimizer'], load_gradients=load_gradients)
		res.optimizer.connect(res, 'w_out', 'b_out', 'embedding_matrix')

		return res
