from utility.utility import search_subclasses


class Model: ...


class Parameterless:
	def to_obj(self):
		return self.__class__.__name__

	@staticmethod
	def from_obj(obj):
		found = search_subclasses(Parameterless, obj)

		if found:
			return found()

		raise KeyError('Class not found')


class Optimizable:
	def drop_gradient(self):
		self.optimizer.drop_gradient()

	def descent(self):
		self.optimizer.descent()


class NoDropout:
	def predict(self, x):
		return self.forward(x)
