from universal import NoDropout

from parts.perceptron import Perceptron
from parts.layer_norm import LayerNorm
from parts.single_head_attention import SingleHeadAttention
from parts.multi_head_attention import MultiHeadAttention


class Residual(NoDropout):
    def __init__(self, *models):
        self.models = models

    def forward(self, x):
        self.x = x
        self.y = self.models[0].forward(x)

        for model in self.models[1:]:
            self.y = model.forward(self.y)

        return x + self.y

    def backward(self, d_in):
        d_out = self.models[-1].backward(d_in)

        for model in self.models[-2::-1]:
            d_out = model.backward(d_out)

        return d_in + d_out

    def drop_gradient(self):
        for model in self.models:
            model.drop_gradient()

    def descent(self):
        for model in self.models:
            model.descent()

    def to_obj(self, save_gradients=False):
        return {
            'models': [(model.__class__.__name__, model.to_obj(save_gradients=save_gradients)) for model in self.models]
        }

    @staticmethod
    def from_obj(obj, load_gradients=True):
        models = [globals()[model[0]].from_obj(model[1], load_gradients=load_gradients) for model in obj['models']]

        return Residual(*models)
