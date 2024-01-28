import torch
import torch.nn.functional as F
import kornia

import matplotlib.pyplot as plt
# 默认图片都是0到1的小数，所以要乘255
def cross_entropy(predict, target, bins=torch.linspace(0, 255, 256), bandwidth=torch.tensor(1),eps=1e-8):
    target = (target*255.0).flatten().unsqueeze(0)
    predict = (predict*255.0).flatten().unsqueeze(0)
    h1 = kornia.enhance.histogram(target, bins, bandwidth=bandwidth).flatten() + eps
    h2 = kornia.enhance.histogram(predict, bins, bandwidth=bandwidth).flatten() + eps
    return torch.sum(h1 * torch.log2(h1 / (h2 + eps)))

cross_entropy_loss = cross_entropy
