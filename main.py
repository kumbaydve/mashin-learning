import torch

from constants import DTYPE
from utility import get_batch_range, get_slice

from activations import Linear, ReLU, Sigmoid, SiLU, Softplus, GELU, Softmax
from losses import MAE, MSE, CrossEntropy

from olegus_ffn import OlegusFFN
from layer_ffn import LayerFFN

olegus = OlegusFFN(
	LayerFFN(2, 4, Softplus()),
	LayerFFN(4, 2, Softmax()),
	CrossEntropy()
)

x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=DTYPE)
y = torch.tensor([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=DTYPE)

epochs = 10_000
batch_size = 1
alpha = 0.1

for epoch in range(1, epochs + 1):
	for batch in get_batch_range(x, batch_size):
		ixs = get_slice(batch, batch_size)

		olegus.drop_gradient()
		olegus.forward(x[ixs, :])
		olegus.backward(y[ixs, :])
		olegus.descent(alpha)

for batch in get_batch_range(x, batch_size):
	ixs = get_slice(batch, batch_size)

	print('input', x[ixs, :].tolist())
	print('output', olegus.forward(x[ixs, :]).tolist())
	print('loss', olegus.get_loss(y[ixs, :]))
	print()
