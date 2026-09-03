def get_batch_range(x, batch_size):
	return range(1, x.shape[0] // batch_size + 1)


def get_slice(batch, batch_size):
	return slice((batch - 1) * batch_size, batch * batch_size)


def search_subclasses(of_class, name):
	for subclass in of_class.__subclasses__():
		if subclass.__name__ == name:
			return subclass
		else:
			found = search_subclasses(subclass, name)

			if found:
				return found

	return None
