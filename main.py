import torch

from constants import FLOAT, INT
from olegus import Olegus
from oleguses.olegus_transformer import OlegusTransformer
from parts.activations import SiLU, Softmax, GELU, Linear
from parts.layer_norm import LayerNorm
from parts.losses import CrossEntropy
from parts.multi_head_attention import MultiHeadAttention
from parts.optimizers import Adam, SGD
from parts.perceptron import Perceptron
from parts.residual import Residual
from parts.sequence import Sequence
from parts.transformer_layer import TransformerLayer

from utility.mnist_utility import get_x_y, mnist_tester
from utility.transformer_utility import get_heads


embedding_d = 128
head_d = 32
seq_len = 16
perceptron_d = 256

'''olegus = Olegus(
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
)'''

#[0.1494, 0.0824, 0.0517, 0.0420, 0.0374, 0.0348, 0.0332, 0.0319, 0.0309, 0.0302]
#[0.1183, 0.0701, 0.0494, 0.0420, 0.0382, 0.0357, 0.0338, 0.0324, 0.0311, 0.0301]

#[0.1576, 0.1046, 0.0718, 0.0549, 0.0462, 0.0412, 0.0378, 0.0355, 0.0337, 0.0323]
#[0.1235, 0.0762, 0.0559, 0.0456, 0.0398, 0.0359, 0.0332, 0.0312, 0.0297, 0.0286]

#[0.1690, 0.1208, 0.0906, 0.0769, 0.0637, 0.0543, 0.0482, 0.0435, 0.0401, 0.0374]
#[0.1660, 0.0999, 0.0680, 0.0522, 0.0442, 0.0392, 0.0357, 0.0332, 0.0313, 0.0297]

'''olegus = Olegus(
	Sequence(
		Perceptron(28*28, 10, SiLU(), Adam(0.001, 0.9, 0.999)),
		Residual(Perceptron(10, 10, SiLU(), Adam(0.001, 0.9, 0.999))),
		Residual(Perceptron(10, 10, SiLU(), Adam(0.001, 0.9, 0.999))),
		Residual(Perceptron(10, 10, SiLU(), Adam(0.001, 0.9, 0.999))),
		Residual(Perceptron(10, 10, SiLU(), Adam(0.001, 0.9, 0.999))),
		Residual(Perceptron(10, 10, SiLU(), Adam(0.001, 0.9, 0.999))),
		Residual(Perceptron(10, 10, SiLU(), Adam(0.001, 0.9, 0.999))),
		#Perceptron(10, 10, SiLU(), Adam(0.001, 0.9, 0.999)),
		#Perceptron(10, 10, SiLU(), Adam(0.001, 0.9, 0.999)),
		#Perceptron(10, 10, SiLU(), Adam(0.001, 0.9, 0.999)),
		#Perceptron(10, 10, SiLU(), Adam(0.001, 0.9, 0.999)),
		#Perceptron(10, 10, SiLU(), Adam(0.001, 0.9, 0.999)),
		#Perceptron(10, 10, SiLU(), Adam(0.001, 0.9, 0.999)),
		Perceptron(10, 10, Softmax(), Adam(0.001, 0.9, 0.999)),
	),
	loss_function=CrossEntropy()
)'''

'''olegus = Olegus(
	Sequence(
		TransformerLayer(32, 8, 128, 16, Adam, (0.001, 0.9, 0.999)),
		TransformerLayer(32, 8, 128, 16, Adam, (0.001, 0.9, 0.999))
	),
	loss_function=CrossEntropy()
)'''

#olegus = Olegus.load('olegus_tr.json')

#x_train, y_train = get_x_y('datasets/mnist_train.csv')
#x_test, y_test = get_x_y('datasets/mnist_test.csv')

#olegus.train(10, 1_000, x_train, y_train, x_test, y_test, tester=mnist_tester)

#olegus.save('olegus_tr.json', save_gradients=True)


dictionary = ['a', 'b', 'c', 'd']

olegus = OlegusTransformer(dictionary, 4, 2, 8, 3, 2, Adam, (0.01, 0.9, 0.999))

ixs = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=INT)

print(olegus.forward(ixs[0, :], 1))
print(olegus.forward(ixs[1, :], 1))

for _ in range(100):
    for i in range(2):
        olegus.drop_gradient()
        olegus.forward(ixs[i, :], 1)
        olegus.store_expected(torch.tensor([0]) if i == 1 else torch.tensor([3]))
        olegus.backward()
        olegus.descent()

print(olegus.forward(ixs[0, :], 1))
print(olegus.forward(ixs[1, :], 1))
