# EINSUM


import torch
import math

from parts.activations import Softmax
from parts.optimizers import Optimizer
from universal import Optimizable, NoDropout, Model


class SingleHeadAttention(Model, Optimizable, NoDropout):
	def __init__(self, embedding_d, head_d, seq_len, optimizer):
		self.embedding_d = embedding_d
		self.head_d = head_d
		self.seq_len = seq_len

		self.w_q = torch.randn((embedding_d, head_d)) * math.sqrt(2 / (embedding_d + head_d))
		self.w_k = torch.randn((embedding_d, head_d)) * math.sqrt(2 / (embedding_d + head_d))
		self.w_v = torch.randn((embedding_d, head_d)) * math.sqrt(2 / (embedding_d + head_d))
		self.activation = Softmax()
		self.optimizer = optimizer

		self.score_mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).to(torch.bool)
		self.sqrt_head_d = 1 / math.sqrt(self.head_d)

		self.optimizer.connect(self, 'w_q', 'w_k', 'w_v')

	def forward(self, x):
		self.x = x
		self.q = x @ self.w_q # bse, eh -> bsh
		self.k = x @ self.w_k # bse, eh -> bsh
		self.v = x @ self.w_v # bse, eh -> bsh

		scores = self.q @ self.k.mT * self.sqrt_head_d
		scores.masked_fill_(self.score_mask, -torch.inf)

		self.attention = self.activation.forward(scores)

		return self.attention @ self.v

	def backward(self, d_in):
		d_activation = self.activation.backward(d_in @ self.v.mT) * self.sqrt_head_d
		d_activation.masked_fill_(self.score_mask, 0)

		d_q = d_activation @ self.k
		d_k = (self.q.mT @ d_activation).mT
		d_v = self.attention.mT @ d_in

		self.optimizer.backward(
			w_q = torch.einsum('...es,...sh->eh', self.x.mT, d_q), # bes, bsh -> eh
			w_k = torch.einsum('...es,...sh->eh', self.x.mT, d_k), # bes, bsh -> eh
			w_v = torch.einsum('...es,...sh->eh', self.x.mT, d_v) # bes, bsh -> eh
		)

		return d_v @ self.w_v.mT + d_k @ self.w_k.mT + d_q @ self.w_q.mT

	def to_obj(self, save_gradients=False):
		return {
			'name': self.__class__.__name__,
			'embedding_d': self.embedding_d,
			'head_d': self.head_d,
			'seq_len': self.seq_len,
			'w_q': self.w_q.tolist(),
			'w_k': self.w_k.tolist(),
			'w_v': self.w_v.tolist(),
			'optimizer': self.optimizer.to_obj(save_gradients=save_gradients)
		}

	@staticmethod
	def from_obj(obj, load_gradients=True):
		res = SingleHeadAttention(obj['embedding_d'], obj['head_d'], obj['seq_len'], Optimizer.from_obj(obj['optimizer'], load_gradients=load_gradients))

		res.w_q = torch.tensor(obj['w_q'])
		res.w_k = torch.tensor(obj['w_k'])
		res.w_v = torch.tensor(obj['w_v'])

		return res
