from universal import Model


class Sequence(Model):
	def __init__(self, *models):
		self.models = models

	def forward(self, x):
		y = self.models[0].forward(x)

		for model in self.models[1:]:
			y = model.forward(y)

		return y

	def predict(self, x):
		y = self.models[0].predict(x)

		for model in self.models[1:]:
			y = model.predict(y)

		return y

	def backward(self, d_in):
		d_out = self.models[-1].backward(d_in)

		for model in self.models[-2::-1]:
			d_out = model.backward(d_out)

		return d_out

	def drop_gradient(self):
		for model in self.models:
			model.drop_gradient()

	def descent(self):
		for model in self.models:
			model.descent()

	def to_obj(self, save_gradients=False):
		return {
			'name': self.__class__.__name__,
			'models': [model.to_obj(save_gradients=save_gradients) for model in self.models],
		}

	@staticmethod
	def from_obj(obj, load_gradients=True):
		models = [Model.from_obj(model, load_gradients=load_gradients) for model in obj['models']]

		return Sequence(*models)
