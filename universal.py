import json

from utility.utility import search_subclasses


class Model:
	def to_obj(self):
		return {
			'name': self.__class__.__name__
		}

	@staticmethod
	def from_obj(obj, load_gradients=False):
		found = search_subclasses(Model, obj['name'])

		if found:
			if getattr(found, 'from_obj') == Model.from_obj:
				return found()
			else:
				return found.from_obj(obj, load_gradients=load_gradients)

		raise KeyError('Class not found')


class Optimizable:
	def drop_gradient(self):
		self.optimizer.drop_gradient()

	def descent(self):
		self.optimizer.descent()


class HasOptimizableModel:
	def drop_gradient(self):
		self.model.drop_gradient()

	def descent(self):
		self.model.descent()


class HasForwardableModel:
	def forward(self, x):
		return self.model.forward(x)

	def backward(self, d_in):
		return self.model.backward(d_in)


class NoDropout:
	def predict(self, x):
		return self.forward(x)


class Savable:
	def save(self, file_name, save_gradients=False):
		with open(file_name, 'w') as file:
			json.dump(self.to_obj(save_gradients=save_gradients), file)

	@staticmethod
	def load(file_name, load_gradients=True):
		with open(file_name, 'r') as file:
			return Model.from_obj(json.load(file), load_gradients=load_gradients)
