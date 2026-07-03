def binary_cross_entropy(y_true, y_pred, epsilon=1e-7):
    return -(y_true * (y_pred + epsilon).log() + (1 - y_true) * (1 - y_pred + epsilon).log())