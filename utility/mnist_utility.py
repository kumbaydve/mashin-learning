import numpy as np
import torch

from constants import FLOAT, INT


def get_x_y(file_name):
	data = torch.from_numpy(np.genfromtxt(file_name, delimiter=',', skip_header=1)).to(FLOAT)

	x = data[:, 1:] / 255
	nums = data[:, 0].to(INT)

	y = torch.zeros((x.shape[0], 10))
	y[torch.arange(0, x.shape[0]), nums] = 1

	return x, y


def predict(y):
	return torch.argmax(y, dim=-1)


def get_num_of_correct(y, expected):
	predicted = predict(y)
	correct = torch.argmax(expected, dim=-1)

	return torch.sum((predicted == correct).to(INT))


losses = torch.empty(0)
percents = torch.empty(0)

def mnist_tester(olegus, x_test, y_test):
	global losses, percents

	y = olegus.predict(x_test)
	loss = olegus.get_loss(y_test).unsqueeze(0)
	percent = torch.round(get_num_of_correct(y, y_test) / y.shape[0] * 100).unsqueeze(0)

	losses = torch.cat([losses, loss])
	percents = torch.cat([percents, percent])

	print('losses', losses)
	print('percents', percents)
