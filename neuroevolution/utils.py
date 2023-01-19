def mse(y_true, y_pred):
    return ((y_pred - y_true) ** 2).mean()


def accuracy(y_true, y_pred):
    return (y_pred == y_true).mean()
