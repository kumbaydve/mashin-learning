import torch

from universal import Model
from constants import EPSILON


class MAE(Model):
	def forward(self, y, expected):
		return torch.mean(torch.abs(y - expected))

	def backward(self, y, expected):
		return torch.sign(y - expected) / y.numel()


class MSE(Model):
	def forward(self, y, expected):
		return torch.mean(torch.square(y - expected))

	def backward(self, y, expected):
		return 2 * (y - expected) / y.numel()


class CrossEntropy(Model):
	def forward(self, y, expected):
		return torch.mean(-expected * torch.log(y + EPSILON))

	def backward(self, y, expected):
		return -expected / (y + EPSILON) / y.numel()
