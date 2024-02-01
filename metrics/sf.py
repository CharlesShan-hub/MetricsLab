import torch
import kornia
import numpy as np
import matplotlib.pyplot as plt

###########################################################################################

__all__ = [
    'sf',
    'sf_approach_loss',
    'sf_metric'
]

def standard_frequency(tensor,eps=1e-10): # 默认输入的是 0-1 的浮点数
    grad_x = kornia.filters.filter2d(tensor,torch.tensor([[1,  -1]], dtype=torch.float32).unsqueeze(0),padding='valid')
    grad_y = kornia.filters.filter2d(tensor,torch.tensor([[1],[-1]], dtype=torch.float32).unsqueeze(0),padding='valid')
    return torch.sqrt(torch.mean(grad_x**2) + torch.mean(grad_y**2) + eps) * 255.0  # 与 VIFB 统一，需要乘 255

def standard_frequency_loss(tensor):
    return -standard_frequency(tensor)

def sf_metric(A, B, F):
    return standard_frequency(F)

###########################################################################################
