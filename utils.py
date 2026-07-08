from value import Value

def binary_cross_entropy(y_true, y_pred, epsilon=1e-7):
    return -(y_true * (y_pred + epsilon).log() + (1 - y_true) * (1 - y_pred + epsilon).log())

def zero_grad(model):
    for p in model.parameters():
        p.grad = 0

def step(model, lr):
    for p in model.parameters():
        p.data -= lr * p.grad

def predict(model, inputs):
    out = model(inputs)[0]
    return out.data

def one_hot(label, n_outputs):
    outputs = [0.0] * n_outputs
    outputs[label] = 1.0
    return outputs

def softmax(values):
    exps = [value.exp() for value in values]

    total = sum(exps)

    return [exp / total for exp in exps]
