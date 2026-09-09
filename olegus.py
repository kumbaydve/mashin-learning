import torch

import json

from universal import Parameterless
from utility.utility import get_batch_range, get_slice

from parts.perceptron import Perceptron
from parts.layer_norm import LayerNorm
from parts.single_head_attention import SingleHeadAttention
from parts.residual import Residual


class Olegus:
	def __init__(self, *models, loss_function=None):
		self.models = models
		self.loss_function = loss_function

	# processes prepared input with dropout, stores x and y
	def forward(self, x):
		self.x = x
		self.y = self.models[0].forward(x)

		for model in self.models[1:]:
			self.y = model.forward(self.y)

		return self.y

	# processes prepared input without dropout, stores x and y
	def predict(self, x):
		self.x = x
		self.y = self.models[0].predict(x)

		for model in self.models[1:]:
			self.y = model.predict(self.y)

		return self.y

	# returns loss_function, stores loss, forward() or predict() must be called before
	def get_loss(self, expected):
		self.loss = self.loss_function.forward(self.y, expected)
		return self.loss

	# returns input gradient, stores nothing, forward() must be called before
	def backward(self, expected):
		d_in = self.loss_function.backward(self.y, expected)

		for model in self.models[::-1]:
			d_in = model.backward(d_in)

		return d_in

	def drop_gradient(self):
		for model in self.models:
			model.drop_gradient()

	def descent(self):
		for model in self.models:
			model.descent()

	def train(self, epochs, batch_size, x_train, y_train, x_test=None, y_test=None, print_losses=True, print_epochs=True, tester=None):
		test_condition = (x_test is not None) and (y_test is not None) and (not tester)
		if test_condition:
			losses_test = torch.empty(epochs)

		for epoch in range(1, epochs + 1):
			if print_epochs:
				print('EPOCH', epoch)

			for batch in get_batch_range(x_train, batch_size):
				ixs = get_slice(batch, batch_size)

				self.drop_gradient()
				self.forward(x_train[ixs, :])
				self.backward(y_train[ixs, :])
				self.descent()

			if (x_test is not None) and (y_test is not None):
				if tester:
					tester(self, x_test, y_test)
				else:
					self.predict(x_test)
					losses_test[epoch - 1] = self.get_loss(y_test)

					if print_losses:
						print('losses', losses_test)

		if test_condition:
			return losses_test

	def to_obj(self, save_gradients=False):
		res = {
			'models': [(model.__class__.__name__, model.to_obj(save_gradients=save_gradients)) for model in self.models],
			'loss_function': self.loss_function.to_obj()
		}

		return res

	@staticmethod
	def from_obj(obj, load_gradients=True):
		models = [globals()[model[0]].from_obj(model[1], load_gradients=load_gradients) for model in obj['models']]
		loss_function = Parameterless.from_obj(obj['loss_function'])

		return Olegus(*models, loss_function=loss_function)

	def save(self, file_name, save_gradients=False):
		with open(file_name, 'w') as file:
			json.dump(self.to_obj(save_gradients=save_gradients), file)

	@staticmethod
	def load(file_name, load_gradients=True):
		with open(file_name, 'r') as file:
			return Olegus.from_obj(json.load(file), load_gradients=load_gradients)
