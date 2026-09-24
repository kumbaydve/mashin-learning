import torch

from universal import Model
from constants import EPSILON


class Optimizer(Model):
	def connect(self, obj, *parameters):
		self.obj = obj

		if not hasattr(self, 'gradients'):
			self.gradients = dict()

			for parameter in parameters:
				self.gradients[parameter] = torch.zeros_like(getattr(obj, parameter))

			self.init_additional()

	def init_additional(self):
		pass

	def backward(self, **gradients):
		for parameter in gradients:
			if isinstance(gradients[parameter], tuple):
				self.gradients[parameter][gradients[parameter][0]] += gradients[parameter][1]
			else:
				self.gradients[parameter] += gradients[parameter]

	def drop_gradient(self):
		for parameter in self.gradients:
			self.gradients[parameter] = torch.zeros_like(self.gradients[parameter])

	def decrease_by(self, parameter, by):
		setattr(self.obj, parameter, getattr(self.obj, parameter) - by)

	def save_gradients(self, res):
		res['gradients'] = dict()

		for parameter in self.gradients:
			res['gradients'][parameter] = self.gradients[parameter].tolist()

	@staticmethod
	def load_gradients(res, obj):
		if 'gradients' not in obj:
			raise KeyError('Gradients are not found')

		res.gradients = dict()

		for parameter in obj['gradients']:
			res.gradients[parameter] = torch.tensor(obj['gradients'][parameter])


class SGD(Optimizer):
	def __init__(self, alpha):
		self.alpha = alpha

	def descent(self):
		for parameter in self.gradients:
			self.decrease_by(parameter, self.alpha * self.gradients[parameter])

	def to_obj(self, save_gradients=False):
		res = {
			'name': self.__class__.__name__,
			'alpha': self.alpha
		}

		if save_gradients:
			self.save_gradients(res)

		return res

	@staticmethod
	def from_obj(obj, load_gradients=True):
		res = SGD(obj['alpha'])

		if load_gradients:
			Optimizer.load_gradients(res, obj)

		return res


class Step(Optimizer):
	def __init__(self, alpha, d, r):
		self.alpha = alpha
		self.d = d
		self.r = r

		self.t = 1

	def descent(self):
		for parameter in self.gradients:
			self.decrease_by(parameter, self.alpha * self.d ** (self.t // self.r) * self.gradients[parameter])

	def to_obj(self, save_gradients=False):
		res = {
			'name': self.__class__.__name__,
			'alpha': self.alpha,
			'd': self.d,
			'r': self.r
		}

		if save_gradients:
			self.save_gradients(res)
			res['t'] = self.t

		return res

	@staticmethod
	def from_obj(obj, load_gradients=True):
		res = Step(obj['alpha'], obj['d'], obj['r'])

		if load_gradients:
			Optimizer.load_gradients(res, obj)
			res.t = obj['t']

		return res


class Adam(Optimizer):
	def __init__(self, alpha, beta_1, beta_2):
		self.alpha = alpha
		self.beta_1 = beta_1
		self.beta_2 = beta_2

		self.t = 1

	def init_additional(self):
		self.means = dict()
		self.variances = dict()

		for parameter in self.gradients:
			self.means[parameter] = torch.zeros_like(self.gradients[parameter])
			self.variances[parameter] = torch.zeros_like(self.gradients[parameter])

	def descent(self):
		for parameter in self.gradients:
			self.means[parameter] = self.beta_1 * self.means[parameter] + (1 - self.beta_1) * self.gradients[parameter]
			self.variances[parameter] = self.beta_2 * self.variances[parameter] + (1 - self.beta_2) * torch.square(self.gradients[parameter])

			mean_corrected = self.means[parameter] / (1 - self.beta_1 ** self.t)
			variance_corrected = self.variances[parameter] / (1 - self.beta_2 ** self.t)

			self.decrease_by(parameter, self.alpha * mean_corrected / (torch.sqrt(variance_corrected) + EPSILON))

		self.t += 1

	def to_obj(self, save_gradients=False):
		res = {
			'name': self.__class__.__name__,
			'alpha': self.alpha,
			'beta_1': self.beta_1,
			'beta_2': self.beta_2
		}

		if save_gradients:
			self.save_gradients(res)

			res['means'] = dict()
			res['variances'] = dict()

			for parameter in self.gradients:
				res['means'][parameter] = self.means[parameter].tolist()
				res['variances'][parameter] = self.variances[parameter].tolist()

			res['t'] = self.t

		return res

	@staticmethod
	def from_obj(obj, load_gradients=True):
		res = Adam(obj['alpha'], obj['beta_1'], obj['beta_2'])

		if load_gradients:
			Optimizer.load_gradients(res, obj)

			res.means = dict()
			res.variances = dict()

			for parameter in obj['gradients']:
				res.means[parameter] = torch.tensor(obj['means'][parameter])
				res.variances[parameter] = torch.tensor(obj['variances'][parameter])

			res.t = obj['t']

		return res
