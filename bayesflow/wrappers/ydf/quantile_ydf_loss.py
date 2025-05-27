import numpy as np
import ydf
from functools import partial


def loss_pinball(labels, predictions, weights, tau):
    residual = labels - predictions
    return np.mean(np.where(residual >= 0, tau * residual, (tau - 1) * residual))


def initial_predictions_pinball(labels, _, tau):
    return np.quantile(labels, tau)


def grad_pinball(labels, predictions, tau):
    residual = labels - predictions
    gradient = np.where(residual >= 0, -tau, 1 - tau)
    return gradient


def hessian_pinball(labels, predictions, tau):
    return -np.ones_like(predictions)


def gradient_and_hessian_pinball(labels, predictions, tau):
    return [grad_pinball(labels, predictions, tau), hessian_pinball(labels, predictions, tau)]


# Construct the loss object.
def make_quantile_loss(tau):
    pinball_custom_loss = ydf.RegressionLoss(
        initial_predictions=partial(initial_predictions_pinball, tau=tau),
        gradient_and_hessian=partial(gradient_and_hessian_pinball, tau=tau),
        loss=partial(loss_pinball, tau=tau),
        activation=ydf.Activation.IDENTITY,
    )
    return pinball_custom_loss
