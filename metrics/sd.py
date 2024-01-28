import torch
import kornia
import numpy as np
import matplotlib.pyplot as plt

def standard_deviation(tensor):
    return torch.sqrt(torch.mean((tensor - tensor.mean())**2))

def standard_deviation_loss(tensor):
    return -standard_deviation(tensor)

