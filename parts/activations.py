import torch

from universal import Parameterless


class Linear(Parameterless):
	def forward(self, x):
		return x

	def backward(self, d_in):
		return d_in


class ReLU(Parameterless):
	def forward(self, x):
		self.x = x
		return torch.maximum(torch.tensor(0), x)

	def backward(self, d_in):
		return torch.where(self.x > 0, d_in, 0)


class Sigmoid(Parameterless):
	def forward(self, x):
		self.s = 1 / (1 + torch.exp(-x))
		return self.s

	def backward(self, d_in):
		return d_in * self.s * (1 - self.s)


class SiLU(Parameterless):
	def forward(self, x):
		self.x = x
		self.s = 1 / (1 + torch.exp(-x))
		self.xs = x * self.s
		return self.xs

	def backward(self, d_in):
		return d_in * self.s * (1 + self.x - self.xs)


class Softplus(Parameterless):
	def forward(self, x):
		self.exp = torch.exp(x)
		return torch.log(1 + self.exp)

	def backward(self, d_in):
		return d_in * self.exp / (1 + self.exp)


class GELU(Parameterless):
	def forward(self, x):
		self.x = x
		self.x_plus_x_cubed = 0.7978845608 * (x + 0.044715 * x ** 3)
		return 0.5 * x * (1 + torch.tanh(self.x_plus_x_cubed))

	def backward(self, d_in):
		return d_in * 0.5 * ((1 + torch.tanh(self.x_plus_x_cubed)) + self.x / (torch.cosh(self.x_plus_x_cubed) ** 2) * 0.7978845608 * (1 + 0.134145 * self.x ** 2))


class Softmax(Parameterless):
	def forward(self, x):
		exp = torch.exp(x - torch.amax(x, dim=-1, keepdim=True))
		self.s = exp / torch.sum(exp, dim=-1, keepdim=True)
		return self.s

	def backward(self, d_in):
		return self.s * (d_in - torch.sum(d_in * self.s, dim=-1, keepdim=True))
