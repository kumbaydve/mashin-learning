from olegus import Olegus
from parts.activations import SiLU, Softmax
from parts.losses import CrossEntropy
from parts.optimizers import Adam
from parts.single_layer_perceptron import SingleLayerPerceptron
from utility.mnist_utility import get_x_y, mnist_tester


olegus = Olegus(
	SingleLayerPerceptron(28 * 28, 20, SiLU(), Adam(0.000_1, 0.9, 0.999), dropout_p=0.1),
	SingleLayerPerceptron(20, 10, SiLU(), Adam(0.000_01, 0.9, 0.999), dropout_p=0.5),
	SingleLayerPerceptron(10, 10, Softmax(), Adam(0.001, 0.9, 0.999)),
	loss_function=CrossEntropy()
)

x_train, y_train = get_x_y('datasets/mnist_train.csv')
x_test, y_test = get_x_y('datasets/mnist_test.csv')

olegus.train(30, 1_000, x_train, y_train, x_test, y_test, tester=mnist_tester)

olegus.save('olegus_mnist.json', save_gradients=True)
