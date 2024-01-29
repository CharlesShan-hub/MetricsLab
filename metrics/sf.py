import torch
import kornia
import numpy as np
import matplotlib.pyplot as plt

def standard_frequency(tensor,eps=1e-8):
    grad_x = kornia.filters.filter2d(tensor,torch.tensor([[1,  -1]], dtype=torch.float32).unsqueeze(0),padding='valid')
    grad_y = kornia.filters.filter2d(tensor,torch.tensor([[1],[-1]], dtype=torch.float32).unsqueeze(0),padding='valid')
    return torch.sqrt(torch.mean(grad_x**2) + torch.mean(grad_y**2) + eps)

def standard_frequency_loss(tensor):
    return -standard_frequency(tensor)
