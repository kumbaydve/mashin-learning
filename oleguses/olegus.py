import torch

from universal import Model, HasOptimizableModel, Savable
from utility.utility import get_batch_range, get_slice


class Olegus(Model, HasOptimizableModel, Savable):
	def __init__(self, model, loss_function=None):
		self.model = model
		self.loss_function = loss_function

	def forward(self, x):
		self.y = self.model.forward(x)
		return self.y

	def predict(self, x):
		self.y = self.model.predict(x)
		return self.y

	def get_loss(self, expected):
		return self.loss_function.forward(self.y, expected)

	def backward(self, expected):
		d_in = self.loss_function.backward(self.y, expected)

		return self.model.backward(d_in)

	def train(self, epochs, batch_size, x_train, y_train, x_test=None, y_test=None, print_losses=True, print_epochs=True, tester=None):
		test_condition = (x_test is not None) and (y_test is not None) and (not tester)
		if test_condition:
			losses_test = torch.empty(epochs)

		for epoch in range(1, epochs + 1):
			if print_epochs:
				print('EPOCH', epoch)

			shuffled_ixs = torch.randperm(x_train.shape[0])

			for batch in get_batch_range(x_train, batch_size):
				ixs = shuffled_ixs[get_slice(batch, batch_size)]

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
		return {
			'name': self.__class__.__name__,
			'model': self.model.to_obj(save_gradients=save_gradients),
			'loss_function': self.loss_function.to_obj()
		}

	@staticmethod
	def from_obj(obj, load_gradients=True):
		model = Model.from_obj(obj['model'], load_gradients=load_gradients)
		loss_function = Model.from_obj(obj['loss_function'])

		return Olegus(model, loss_function=loss_function)
