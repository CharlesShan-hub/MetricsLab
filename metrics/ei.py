import torch
import kornia
import numpy as np
import matplotlib.pyplot as plt

def edge_intensity(tensor,border_type='replicate',eps=1e-10): # 默认输入的是 0-1 的浮点数
    grad_x = kornia.filters.filter2d(tensor,torch.tensor([[ 1,  2,  1],[ 0,  0,  0],[-1, -2, -1]], dtype=torch.float64).unsqueeze(0),border_type=border_type)
    grad_y = kornia.filters.filter2d(tensor,torch.tensor([[ 1,  0, -1],[ 2,  0, -2],[ 1,  0, -1]], dtype=torch.float64).unsqueeze(0),border_type=border_type)
    s = torch.sqrt(grad_x ** 2 + grad_y ** 2 + eps)
    return torch.mean(s) * 255 # 与 VIFB 统一，需要乘 255

def edge_intensity_loss(tensor):
    return -edge_intensity(tensor)

def ei_metric(imgA,imgB,imgF):
    return edge_intensity(imgF)

def main():
    from PIL import Image
    from torchvision import transforms
    import torchvision.transforms.functional as TF

    tensor = TF.to_tensor(Image.open('../imgs/TNO/fuse/U2Fusion/9.bmp')).unsqueeze(0)
    tensor = torch.clamp(torch.mul(tensor, 255), 0, 255).to(torch.float64)

    tensor = edge_intensity(tensor)
    print(tensor)
if __name__ == '__main__':
    main()