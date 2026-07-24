class OlegusFFN:
	def __init__(self, *structure):
		self.layers = structure[:-1]
		self.loss = structure[-1]

	def forward(self, x):
		self.y = x

		for layer in self.layers:
			self.y = layer.forward(self.y)

		return self.y

	def get_loss(self, expected):
		return self.loss.forward(self.y, expected)

	def backward(self, expected):
		d_in = self.loss.backward(self.y, expected)

		for layer in self.layers[::-1]:
			d_in = layer.backward(d_in)

		return d_in

	def drop_gradient(self):
		for layer in self.layers:
			layer.drop_gradient()

	def descent(self, alpha):
		for layer in self.layers:
			layer.descent(alpha)
