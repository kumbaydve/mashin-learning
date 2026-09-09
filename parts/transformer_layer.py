'''import math
import torch

from parts.single_head_attention import SingleHeadAttention
from parts.layer_norm import LayerNorm
from parts.optimizers import Optimizer
from parts.perceptron import Perceptron
from parts.activations import GELU, Linear


class Transformer:
	def __init_bad__(self, embedding_d, head_d, perceptron_d, seq_len, optimizer, head_optimizer_class, head_optimizer_parameters, layer_norm_optimizer_class, layer_norm_optimizer_parameters, perceptron_optimizer_class, perceptron_optimizer_parameters, from_obj=False):
		self.embedding_d = embedding_d
		self.head_d = head_d
		self.perceptron_d = perceptron_d
		self.head_n = embedding_d // head_d
		self.seq_len = seq_len
		self.optimizer = optimizer

		if not from_obj:
			self.heads = [SingleHeadAttention(embedding_d, head_d, seq_len, head_optimizer_class(*head_optimizer_parameters)) for _ in range(self.head_n)]

			self.w_out = torch.randn(embedding_d, embedding_d) * math.sqrt(1 / embedding_d)

			self.norm_1 = LayerNorm(embedding_d, layer_norm_optimizer_class(*layer_norm_optimizer_parameters))
			self.perceptron_1 = Perceptron(embedding_d, perceptron_d, GELU(), perceptron_optimizer_class(*perceptron_optimizer_parameters))
			self.perceptron_2 = Perceptron(perceptron_d, embedding_d, Linear(), perceptron_optimizer_class(*perceptron_optimizer_parameters))
			self.norm_2 = LayerNorm(embedding_d, layer_norm_optimizer_class(*layer_norm_optimizer_parameters))

		self.optimizer.connect(self, 'w_out')

	def __init__(self, embedding_d, head_d, perceptron_d, seq_len, optimizer, layer_norm_optimizer_class, layer_norm_optimizer_parameters, perceptron_optimizer_class, perceptron_optimizer_parameters, from_obj=False):
		self.embedding_d = embedding_d
		self.head_d = head_d
		self.perceptron_d = perceptron_d
		self.head_n = embedding_d // head_d
		self.seq_len = seq_len
		self.optimizer = optimizer

		if not from_obj:
			self.heads = [SingleHeadAttention(embedding_d, head_d, seq_len, head_optimizer_class(*head_optimizer_parameters)) for _ in range(self.head_n)]

			self.w_out = torch.randn(embedding_d, embedding_d) * math.sqrt(1 / embedding_d)

			self.norm_1 = LayerNorm(embedding_d, layer_norm_optimizer_class(*layer_norm_optimizer_parameters))
			self.perceptron_1 = Perceptron(embedding_d, perceptron_d, GELU(), perceptron_optimizer_class(*perceptron_optimizer_parameters))
			self.perceptron_2 = Perceptron(perceptron_d, embedding_d, Linear(), perceptron_optimizer_class(*perceptron_optimizer_parameters))
			self.norm_2 = LayerNorm(embedding_d, layer_norm_optimizer_class(*layer_norm_optimizer_parameters))

		self.optimizer.connect(self, 'w_out')

	def forward_bad(self, x):
		self.x = x
		head_outputs = []

		for head in self.heads:
			head_outputs.append(head.forward(x))

		self.concat = torch.concatenate(head_outputs, dim=-1)
		multi_head = self.concat @ self.w_out
		after_multi_head = x + multi_head

		after_norm_1 = self.norm_1.forward(after_multi_head)
		perceptron_1_output = self.perceptron_1.forward(after_norm_1)
		perceptron_2_output = self.perceptron_2.forward(perceptron_1_output)
		after_perceptron = after_norm_1 + perceptron_2_output

		self.y = self.norm_2.forward(after_perceptron)

		return self.y

	def forward(self, x):
		self.x = x
		head_outputs = []

		for head in self.heads:
			head_outputs.append(head.forward(x))

		self.concat = torch.concatenate(head_outputs, dim=-1)
		multi_head = self.concat @ self.w_out
		after_multi_head = x + multi_head

		after_norm_1 = self.norm_1.forward(after_multi_head)
		perceptron_1_output = self.perceptron_1.forward(after_norm_1)
		perceptron_2_output = self.perceptron_2.forward(perceptron_1_output)
		after_perceptron = after_norm_1 + perceptron_2_output

		self.y = self.norm_2.forward(after_perceptron)

		return self.y

	def backward(self, d_in):
		d_norm_2 = self.norm_2.backward(d_in)
		d_perceptron_2 = self.perceptron_2.backward(d_norm_2)
		d_perceptron_1 = self.perceptron_1.backward(d_perceptron_2)
		d_after_norm_1 = self.norm_1.backward(d_perceptron_1 + d_norm_2)

		d_concat = d_after_norm_1 @ self.w_out.T
		self.optimizer.backward(
			w_out = self.concat.T @ d_after_norm_1
		)

		d_heads = torch.split(d_concat, split_size_or_sections=self.head_d, dim=-1)
		d_x = d_after_norm_1

		for i in range(self.head_n):
			d_x += self.heads[i].backward(d_heads[i])

		return d_x

	def drop_gradient(self):
		self.norm_2.drop_gradient()
		self.perceptron_2.drop_gradient()
		self.perceptron_1.drop_gradient()
		self.norm_1.drop_gradient()

		for head in self.heads:
			head.drop_gradient()

		self.optimizer.drop_gradient()

	def descent(self):
		self.norm_2.descent()
		self.perceptron_2.descent()
		self.perceptron_1.descent()
		self.norm_1.descent()

		for head in self.heads:
			head.descent()

		self.optimizer.descent()

	def to_obj(self, save_gradients=False):
		return {
			'embedding_d': self.embedding_d,
			'head_d': self.head_d,
			'perceptron_d': self.perceptron_d,
			'seq_len': self.seq_len,
			'heads': [head.to_obj(save_gradients=save_gradients) for head in self.heads],
			'w_out': self.w_out.tolist(),
			'norm_1': self.norm_1.to_obj(save_gradients=save_gradients),
			'perceptron_1': self.perceptron_1.to_obj(save_gradients=save_gradients),
			'perceptron_2': self.perceptron_2.to_obj(save_gradients=save_gradients),
			'norm_2': self.norm_2.to_obj(save_gradients=save_gradients),
			'optimizer': self.optimizer.to_obj(save_gradients=save_gradients),
		}

	@staticmethod
	def from_obj(obj, load_gradients=True):
		res = Transformer(obj['embedding_d'], obj['head_d'], obj['perceptron_d'], obj['seq_len'], Optimizer.from_obj(obj['optimizer'], load_gradients=load_gradients), None, None, None, None, None, None, from_obj=True)

		res.heads = [SingleHeadAttention.from_obj(head, load_gradients=load_gradients) for head in obj['heads']]
		res.w_out = torch.tensor(obj['w_out'])
		res.norm_1 = LayerNorm.from_obj(obj['norm_1'], load_gradients=load_gradients)
		res.perceptron_1 = Perceptron.from_obj(obj['perceptron_1'], load_gradients=load_gradients)
		res.perceptron_2 = Perceptron.from_obj(obj['perceptron_2'], load_gradients=load_gradients)
		res.norm_2 = LayerNorm.from_obj(obj['norm_2'], load_gradients=load_gradients)
'''