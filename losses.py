import torch


class Loss:
	def forward(self, y, expected):
		return self.function(y, expected).item()

	def backward(self, y, expected):
		return self.derivative(y, expected) / y.numel()

	def function(self, y, expected):
		raise AttributeError('Function not defined')

	def derivative(self, y, expected):
		raise AttributeError('Derivative not defined')


class MAE(Loss):
	def function(self, y, expected):
		return torch.mean(torch.abs(y - expected))

	def derivative(self, y, expected):
		return torch.sign(y - expected)


class MSE(Loss):
	def function(self, y, expected):
		return torch.mean(torch.square(y - expected))

	def derivative(self, y, expected):
		return 2 * (y - expected)


class CrossEntropy(Loss):
	def function(self, y, expected):
		return torch.mean(-expected * torch.log(y))

	def derivative(self, y, expected):
		return -expected / y
