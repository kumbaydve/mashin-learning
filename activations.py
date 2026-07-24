import torch


class Activation:
	def forward(self, x):
		self.x = x
		return self.function()

	def function(self):
		raise AttributeError('Function not defined')


class Linear(Activation):
	def function(self):
		return self.x

	def backward(self, d_in):
		return d_in * torch.ones_like(self.x)


class ReLU(Activation):
	def function(self):
		return torch.maximum(torch.tensor(0), self.x)

	def backward(self, d_in):
		return d_in * torch.where(self.x > 0, 1, 0)


class Sigmoid(Activation):
	def function(self):
		self.s = 1 / (1 + torch.exp(-self.x))
		return self.s

	def backward(self, d_in):
		return d_in * self.s * (1 - self.s)


class SiLU(Activation):
	def function(self):
		self.s = 1 / (1 + torch.exp(-self.x))
		return self.x * self.s

	def backward(self, d_in):
		return d_in * self.s * (1 + self.x * (1 - self.s))


class Softplus(Activation):
	def function(self):
		self.exp = torch.exp(self.x)
		return torch.log(1 + self.exp)

	def backward(self, d_in):
		return d_in * self.exp / (1 + self.exp)


class GELU(Activation):
	def function(self):
		self.x_plus_x_cubed = 0.7978845608 * (self.x + 0.044715 * self.x ** 3)
		return 0.5 * self.x * (1 + torch.tanh(self.x_plus_x_cubed))

	def backward(self, d_in):
		return d_in * 0.5 * ((1 + torch.tanh(self.x_plus_x_cubed)) + self.x / (torch.cosh(self.x_plus_x_cubed) ** 2) * 0.7978845608 * (1 + 0.134145 * self.x ** 2))


class Softmax(Activation):
	def function(self):
		self.exp = torch.exp(self.x - torch.amax(self.x, dim=-1, keepdim=True))
		self.s = self.exp / torch.sum(self.exp, dim=-1, keepdim=True)
		return self.s

	def backward(self, d_in):
		return self.s * (d_in - torch.sum(d_in * self.s, dim=-1, keepdim=True))
