import torch
import torch.nn.functional as F

def rmse(y_true, y_pred):
    mse_loss = torch.mean((y_true - y_pred)**2)
    rmse_loss = torch.sqrt(mse_loss)
    return rmse_loss

rmse_loss = rmse

def rmse_metric(imgA,imgB,imgF):
    w0 = w1 = 0.5
    return w0 * rmse(imgA,imgF) + w1 * rmse(imgB,imgF)