from utility.utility import search_subclasses


class Parameterless:
	def to_obj(self):
		return self.__class__.__name__

	@staticmethod
	def from_obj(obj):
		found = search_subclasses(Parameterless, obj)

		if found:
			return found()

		raise KeyError('Class not found')
