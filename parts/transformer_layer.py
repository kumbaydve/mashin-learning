from parts.activations import GELU, Linear
from parts.layer_norm import LayerNorm
from parts.multi_head_attention import MultiHeadAttention
from parts.perceptron import Perceptron
from parts.residual import Residual
from parts.sequence import Sequence
from universal import NoDropout, Model, HasForwardableModel, HasOptimizableModel
from utility.transformer_utility import get_heads


class TransformerLayer(Model, NoDropout, HasForwardableModel, HasOptimizableModel):
	def __init__(self, embedding_d, head_d, perceptron_d, seq_len, optimizer_class, optimizer_parameters, from_obj=False):
		if not from_obj:
			self.model = Sequence(
				Residual(
					MultiHeadAttention(embedding_d, head_d, seq_len, optimizer_class(*optimizer_parameters), get_heads(embedding_d, head_d, seq_len, optimizer_class, optimizer_parameters))
				),
				LayerNorm(embedding_d, optimizer_class(*optimizer_parameters)),
				Residual(Sequence(
					Perceptron(embedding_d, perceptron_d, GELU(), optimizer_class(*optimizer_parameters)),
					Perceptron(perceptron_d, embedding_d, Linear(), optimizer_class(*optimizer_parameters))
				)),
				LayerNorm(embedding_d, optimizer_class(*optimizer_parameters))
			)

	def to_obj(self, save_gradients=False):
		return {
			'name': self.__class__.__name__,
			'model': self.model.to_obj(save_gradients=save_gradients)
		}

	@staticmethod
	def from_obj(obj, load_gradients=True):
		res = TransformerLayer(0, 0, 0, 0, 0, 0, from_obj=True)

		res.model = Model.from_obj(obj['model'], load_gradients=load_gradients)

		return res
