def get_batch_range(x, batch_size):
	return range(1, x.shape[0] // batch_size + 1)

def get_slice(batch, batch_size):
	return slice((batch - 1) * batch_size, batch * batch_size)
