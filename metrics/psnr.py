from kornia.metrics import psnr as psnr_kornia
from kornia.losses import psnr_loss as psnr_loss_kornia
import torch
import torch.nn.functional as F

def psnr(imgA,imgB,imgF,MAX=1): # 改造成 VIFB 提出的用于融合的 PSNR
    # https://jason-chen-1992.weebly.com/home/-peak-single-to-noise-ratio
    MSE1 = torch.mean((imgA - imgF)**2)
    MSE2 = torch.mean((imgB - imgF)**2)
    MSE = (MSE1+MSE2)/2.0
    return 10*torch.log10(MAX**2/MSE) # 图像融合，MSE 不可能等于零

def psnr_loss(imgA,imgB,imgF,MAX=1):
    return -psnr(imgA,imgB,imgF,MAX=MAX)

def psnr_metric(imgA,imgB,imgF):
    # w0 = w1 = 0.5
    # return w0 * psnr_kornia(imgF,imgA,max_val=1) + w1 * psnr_kornia(imgF,imgB,max_val=1)
    # return w0 * psnr_kornia(imgF*255,imgA*255,max_val=255) + w1 * psnr_kornia(imgF*255,imgB*255,max_val=255) # 为了与VIFB 统一
    # 发现原来 VIFB 里边的两个 MAE 竟然在一个 log 里边！所以不能分开算两个 PSNR 再求平均。
    return psnr(imgA,imgB,imgF)