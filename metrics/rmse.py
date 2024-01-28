import torch
import torch.nn.functional as F

def rmse(y_true, y_pred):
    mse_loss = F.mse_loss(y_true, y_pred)
    rmse_loss = torch.sqrt(mse_loss)
    return rmse_loss

rmse_loss = rmse