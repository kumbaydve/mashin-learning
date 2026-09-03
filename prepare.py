import torch


def prepare_poly_reg(num_of_vars, power, input):
	num_of_inputs = num_of_vars * power
	x = torch.empty(input.shape[0], num_of_inputs)

	for sample in range(input.shape[0]):
		for var in range(input.shape[1]):
			val = input[var].item()

			for power in range(power):
				x[sample, var * power + power] = val
				val *= input[var]

	return x
