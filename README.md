OLEGUS
======


### 1. Introduction
This project contains classes and utility, that can be assembled into a trainable model. That model's name is Olegus (always, no matter what you do)

### 2. Get started
1. See what is available in `/parts`. This folder has all the building blocks of the model (like `Perceptron`, `SingleHeadAttention` and `Residual`)
2. Check `/examples`. It should give you brief understanding of how Olegus is created and trained
3. Maybe look at `/utility`

### 3. Training process
1. Load data. Doesn't matter how, just have it as a `torch.tensor`. Dimension `0` must be batch (except for transformer; `einsum` is not integrated yet)
2. Initialize Olegus. Class `Olegus` takes a model as the first argument and a loss function as the second (both instances, not classes: `MSE()`, not `MSE`). Model is just one argument, so for combining many of them, use `Sequence(model_1, model_2, ...)`
3. Start training loop. First, set gradient to `0` with `olegus.drop_gradient()`. Second, call `olegus.forward(data)`. This will store the output in Olegus and allow backpropagation. Third, call `olegus.backward(expected)`, which will add new gradient based on input to already stored gradient. Finally, apply gradient by `olegus.descent()`. Optionally, loss can be queried with `olegus.get_loss(expected)` (works only after `olegus.forward(data)`, just like `olegus.backward(expected)`)
4. Save Olegus. `olegus.to_obj()` will return Python `dict` and `olegus.save(file_name)` will create the `dict` and write it into a file
