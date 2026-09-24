import torch
import math

from universal import Optimizable, Model
from parts.activations import Linear, ReLU, SiLU, Softplus, GELU
from parts.optimizers import Optimizer
from constants import FLOAT


class Perceptron(Model, Optimizable):
	def __init__(self, in_dim, out_dim, activation, optimizer, dropout_p=0):
		self.in_dim = in_dim
		self.out_dim = out_dim

		self.w = torch.randn((in_dim, out_dim))
		self.b = torch.zeros((1, out_dim))
		self.activation = activation
		self.optimizer = optimizer
		self.dropout_p = dropout_p

		if isinstance(activation, (Linear, ReLU, SiLU, Softplus, GELU)):
			self.w *= math.sqrt(2 / in_dim)
		else:
			self.w *= math.sqrt(2 / (in_dim + out_dim))

		self.optimizer.connect(self, 'w', 'b')

	def forward(self, x):
		self.x = x
		y = self.activation.forward(x @ self.w + self.b)

		if self.dropout_p:
			self.mask = (torch.rand_like(y) >= self.dropout_p).to(FLOAT) / (1 - self.dropout_p)
			y *= self.mask

		return y

	def predict(self, x):
		return self.activation.forward(x @ self.w + self.b)

	def backward(self, d_in):
		if self.dropout_p:
			d_z = self.activation.backward(d_in * self.mask)
		else:
			d_z = self.activation.backward(d_in)

		self.optimizer.backward(
			w = self.x.T @ d_z,
			b = torch.sum(d_z, dim=0)
		)

		return d_z @ self.w.T

	def to_obj(self, save_gradients=False):
		return {
			'name': self.__class__.__name__,
			'in_dim': self.in_dim,
			'out_dim': self.out_dim,
			'w': self.w.tolist(),
			'b': self.b.tolist(),
			'activation': self.activation.to_obj(),
			'optimizer': self.optimizer.to_obj(save_gradients=save_gradients),
			'dropout_p': self.dropout_p
		}

	@staticmethod
	def from_obj(obj, load_gradients=True):
		res = Perceptron(obj['in_dim'], obj['out_dim'], Model.from_obj(obj['activation']), Optimizer.from_obj(obj['optimizer'], load_gradients=load_gradients), dropout_p=obj['dropout_p'])

		res.w = torch.tensor(obj['w'])
		res.b = torch.tensor(obj['b'])

		return res
