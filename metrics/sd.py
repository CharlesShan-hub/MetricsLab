import torch
import kornia
import numpy as np
import matplotlib.pyplot as plt

def standard_deviation(tensor): # 默认输入的是 0-1 的浮点数
    return torch.sqrt(torch.mean((tensor - tensor.mean())**2)) * 255.0  # 与 VIFB 统一，需要乘 255

def standard_deviation_loss(tensor):
    return -standard_deviation(tensor)

def sd_metric(imgA, imgB, imgF):
    return standard_deviation(imgF)