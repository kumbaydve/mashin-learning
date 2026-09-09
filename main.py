from olegus import Olegus
from parts.activations import SiLU, Softmax, GELU, Linear
from parts.layer_norm import LayerNorm
from parts.losses import CrossEntropy
from parts.multi_head_attention import MultiHeadAttention
from parts.optimizers import Adam
from parts.perceptron import Perceptron
from parts.residual import Residual

from utility.mnist_utility import get_x_y, mnist_tester
from utility.transformer_utility import get_heads


embedding_d = 128
head_d = 32
seq_len = 16
perceptron_d = 256

olegus = Olegus(
	Residual(
		MultiHeadAttention(embedding_d, head_d, seq_len, Adam(0.001, 0.9, 0.999), get_heads(embedding_d, head_d, seq_len, Adam, (0.001, 0.9, 0.999)))
	),
	LayerNorm(embedding_d, Adam(0.001, 0.9, 0.999)),
	Residual(
		Perceptron(embedding_d, perceptron_d, GELU(), Adam(0.001, 0.9, 0.999)),
		Perceptron(perceptron_d, embedding_d, Linear(), Adam(0.001, 0.9, 0.999))
	),
	LayerNorm(embedding_d, Adam(0.001, 0.9, 0.999)),
	loss_function=CrossEntropy()
)

#x_train, y_train = get_x_y('datasets/mnist_train.csv')
#x_test, y_test = get_x_y('datasets/mnist_test.csv')

#olegus.train(10, 1_000, x_train, y_train, x_test, y_test, tester=mnist_tester)

#olegus.save('olegus_mnist.json', save_gradients=True)
