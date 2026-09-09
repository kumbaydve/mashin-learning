import torch
import math

from parts.optimizers import Optimizer
from parts.single_head_attention import SingleHeadAttention
from universal import Optimizable, NoDropout


class MultiHeadAttention(Optimizable, NoDropout):
	def __init__(self, embedding_d, head_d, seq_len, optimizer, heads):
		self.embedding_d = embedding_d
		self.head_d = head_d
		self.head_n = embedding_d // head_d
		self.seq_len = seq_len
		self.optimizer = optimizer

		self.heads = heads
		self.w_out = torch.randn(embedding_d, embedding_d) * math.sqrt(1 / embedding_d)

		self.optimizer.connect(self, 'w_out')

	def forward(self, x):
		self.x = x

		head_outputs = []

		for head in self.heads:
			head_outputs.append(head.forward(x))

		self.concat = torch.concatenate(head_outputs, dim=-1)
		self.y = self.concat @ self.w_out

		return self.y

	def backward(self, d_in):
		d_concat = d_in @ self.w_out.T

		self.optimizer.backward(
			w_out = self.concat.T @ d_in
		)

		d_heads = torch.split(d_concat, split_size_or_sections=self.head_d, dim=-1)
		d_x = self.heads[0].backward(d_heads[0])

		for i in range(1, self.head_n):
			d_x += self.heads[i].backward(d_heads[i])

		return d_x

	def to_obj(self, save_gradients=False):
		return {
			'embedding_d': self.embedding_d,
			'head_d': self.head_d,
			'seq_len': self.seq_len,
			'heads': [head.to_obj(save_gradients=save_gradients) for head in self.heads],
			'w_out': self.w_out.tolist(),
			'optimizer': self.optimizer.to_obj(save_gradients=save_gradients)
		}

	@staticmethod
	def from_obj(obj, load_gradients=True):
		res = MultiHeadAttention(obj['embedding_d'], obj['head_d'], obj['seq_len'], Optimizer.from_obj(obj['optimizer'], load_gradients=load_gradients), [SingleHeadAttention.from_obj(head, load_gradients=load_gradients) for head in obj['heads']])

		res.w_out = torch.tensor(obj['w_out'])

		return res
