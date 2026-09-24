from universal import NoDropout, Model, HasOptimizableModel


class Residual(Model, NoDropout, HasOptimizableModel):
	def __init__(self, model):
		self.model = model

	def forward(self, x):
		return x + self.model.forward(x)

	def backward(self, d_in):
		return d_in + self.model.backward(d_in)

	def to_obj(self, save_gradients=False):
		return {
			'name': self.__class__.__name__,
			'model': self.model.to_obj(save_gradients=save_gradients)
		}

	@staticmethod
	def from_obj(obj, load_gradients=True):
		model = Model.from_obj(obj['model'], load_gradients=load_gradients)

		return Residual(model)
