import torch
import torch.nn.functional as F
import kornia

import matplotlib.pyplot as plt
# 默认图片都是0到1的小数，所以要乘255
def cross_entropy(target, predict, bandwidth=0.1, eps=1e-10):
    predict = predict.view(1, -1) * 255
    target = target.view(1, -1) * 255
    bins = torch.linspace(0, 255, 256).to(predict.device)
    h1 = kornia.enhance.histogram(target, bins=bins, bandwidth=torch.tensor(bandwidth))
    h2 = kornia.enhance.histogram(predict, bins=bins, bandwidth=torch.tensor(bandwidth))
    mask = (h1 > eps)&( h2 > eps)
    return torch.sum(h1[mask] * torch.log2(h1[mask]/(h2[mask])))

def cross_entropy_loss(target, predict, bandwidth=0.1, eps=1e-20):
    return cross_entropy(target, predict, bandwidth=bandwidth, eps=eps)

def ce_metric(imgA, imgB, imgF):
    w0 = w1 = 0.5
    return w0 * cross_entropy(imgA,imgF) + w1 * cross_entropy(imgB,imgF)
