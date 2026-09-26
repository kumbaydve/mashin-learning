# EINSUM


import torch

from universal import NoDropout, Model, Optimizable
from constants import EPSILON
from parts.optimizers import Optimizer


class LayerNorm(Model, Optimizable, NoDropout):
	def __init__(self, dim, optimizer):
		self.dim = dim
		self.gamma = torch.ones(dim)
		self.beta = torch.zeros(dim)
		self.optimizer = optimizer

		self.optimizer.connect(self, 'gamma', 'beta')

	def forward(self, x):
		self.x = x
		mean = torch.mean(x, dim=-1, keepdim=True)
		var = torch.var(x, dim=-1, keepdim=True)
		self.std = torch.sqrt(var + EPSILON)

		self.norm = (x - mean) / self.std

		return self.gamma * self.norm + self.beta

	def backward(self, d_in): # bse
		self.optimizer.backward(
			gamma = torch.einsum('...se->e', d_in * self.norm), # bse -> e
			beta = torch.einsum('...se->e', d_in) # bse -> e
		)

		d_pre_norm = d_in * self.gamma / self.std

		return d_pre_norm - torch.mean(d_pre_norm, dim=-1, keepdim=True) - (self.norm * torch.sum(d_pre_norm * self.norm, dim=-1, keepdim=True)) / self.x.shape[-1]

	def to_obj(self, save_gradients=False):
		return {
			'name': self.__class__.__name__,
			'dim': self.dim,
			'gamma': self.gamma.tolist(),
			'beta': self.beta.tolist(),
			'optimizer': self.optimizer.to_obj(save_gradients=save_gradients)
		}

	@staticmethod
	def from_obj(obj, load_gradients=True):
		res = LayerNorm(obj['dim'], Optimizer.from_obj(obj['optimizer'], load_gradients=load_gradients))

		res.gamma = torch.tensor(obj['gamma'])
		res.beta = torch.tensor(obj['beta'])

		return res
