import torch
import math

from activations import Linear, ReLU, Sigmoid, SiLU, Softplus, GELU, Softmax


class LayerFFN:
	def __init__(self, in_dim, out_dim, activation):
		self.in_dim = in_dim
		self.out_dim = out_dim

		self.w = torch.randn((in_dim, out_dim))
		self.b = torch.zeros((1, out_dim))
		self.activation = activation

		if isinstance(activation, (Linear, ReLU, SiLU, Softplus, GELU)):
			self.w *= math.sqrt(2 / in_dim)
		else:
			self.w *= math.sqrt(2 / (in_dim + out_dim))

		self.d_w = torch.zeros_like(self.w)
		self.d_b = torch.zeros_like(self.b)

	def forward(self, x):
		self.x = x

		self.z = x @ self.w + self.b

		return self.activation.forward(self.z)

	def backward(self, d_in):
		d_z = self.activation.backward(d_in)

		self.d_w += self.x.T @ d_z
		self.d_b += torch.sum(d_z, dim=0)

		return d_z @ self.w.T

	def drop_gradient(self):
		self.d_w = torch.zeros_like(self.w)
		self.d_b = torch.zeros_like(self.b)

	def descent(self, alpha):
		self.w -= alpha * self.d_w
		self.b -= alpha * self.d_b
