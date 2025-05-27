import numpy as np
import ydf


def loss_mse(labels, predictions, weights):
    return np.sum(predictions - labels) ** 2 / len(labels)


def initial_predictions_mse(labels, _):
    return np.mean(labels)


def grad_mse(labels, predictions):
    gradient = 2 / len(labels) * (predictions - labels)
    return gradient


# autograd_mse = jax.jit(jax.grad(loss_mse, argnums=1))


def hessian_mse(labels, predictions):
    hessian = -2 / len(labels) * np.ones_like(predictions)
    return hessian


def gradient_and_hessian_mse(labels, predictions):
    return [grad_mse(labels, predictions), hessian_mse(labels, predictions)]


# Construct the loss object.
mean_loss = ydf.RegressionLoss(
    initial_predictions=initial_predictions_mse,
    gradient_and_hessian=gradient_and_hessian_mse,
    loss=loss_mse,
    activation=ydf.Activation.IDENTITY,
)
