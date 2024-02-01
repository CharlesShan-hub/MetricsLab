import torch
import kornia
import numpy as np
import matplotlib.pyplot as plt

###########################################################################################

__all__ = [
    'sd',
    'sd_approach_loss',
    'sd_metric'
]

def standard_deviation(tensor): # 默认输入的是 0-1 的浮点数
    return torch.sqrt(torch.mean((tensor - tensor.mean())**2))# * 255.0  # 与 VIFB 统一，需要乘 255

def standard_deviation_loss(tensor):
    return -standard_deviation(tensor)

def sd_metric(A, B, F):
    return standard_deviation(F)

###########################################################################################
