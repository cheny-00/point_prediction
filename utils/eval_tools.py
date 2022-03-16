

import torch
import numpy as np

def ade(pred, target):
    pred, target = pred.clone().detach().cpu(), target.clone().detach().cpu()
    dist = torch.sum(torch.sqrt(torch.sum(torch.pow(pred - target, 2), -1))) / len(pred)
    return dist

# def mean_angle_error():
#
#     angle_error = []
#
#     for i in range(len(X_test)):
#         x_input = X_test[i]
#         x_input = x_input.reshape((1, 2, 6))
#         yhat = model.predict(x_input)
#         yhat = yhat.reshape(2, 1)
#         y_true = y_test[i]
#         angle_hat = np.arctan((x_input[0][1][5] - yhat[1]) / (x_input[0][0][5] - yhat[0]))
#         angle_true = np.arctan((x_input[0][1][5] - y_true[1]) / (x_input[0][0][5] - y_true[0]))
#         error_i = np.absolute(angle_hat - angle_true)
#         angle_error.append(error_i)