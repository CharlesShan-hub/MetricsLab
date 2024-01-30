import torch
import torch.fft
import kornia
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

def _normalize(data):
    max_value = torch.max(data)
    min_value = torch.min(data)
    if max_value == 0 and min_value == 0: return data
    else: newdata = (data - min_value) / (max_value - min_value)
    return newdata * 255
    # return torch.round(newdata * 255) # <- 最终确定就是这一步导致的梯度消失

def gaussian2d(sigma,size=31):
      meshgrid = kornia.create_meshgrid(size, size, normalized_coordinates=False)
      x = meshgrid[0, :, :, 0] - (size - 1) / 2
      y = meshgrid[0, :, :, 1] - (size - 1) / 2
      gaussian_filter = torch.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * torch.tensor([torch.pi], dtype=torch.float32) * sigma**2)
      return gaussian_filter

G1 = gaussian2d(sigma=2,size=31) #  VIFB 这么设置的参数
G2 = gaussian2d(sigma=4,size=31)

def contrast_sensitivity_filtering_Sd(size=None,mode='spatial'):
    # f0 f2 a
    f0 = 15.3870;f1 = 1.3456;a = 0.7622 # A.B. Watson, A.J. Ahumada Jr., A standard model for foveal detection of spatial contrast, Journal of Vision 5 (9) (2005) 717–740.
    
    # kernel size
    if mode == 'frequency': 
        _, _, m, n = size # VIFB 希望频率的 dog 和原图一样大
        m0 = m/30;n0 = n/30 # VIFB里边这么实现的
        m1 = m/2; n1 = n/2  # DoG1
        m2 = m/4; n2 = n/4  # DoG2
        m3 = m/8; n3 = n/8  # DoG3
    elif mode == 'spatial': # 最后证明
        m=n=7 # 我想转换到时域所以 size 需要很小（我加的）
        m0 = m/2; n0 = n/2  # DoG1 
    
    # meshgrid
    #u,v = torch.meshgrid(torch.linspace(-1, 1, n, dtype=torch.float64), torch.linspace(-1, 1, m, dtype=torch.float64))
    #u = u*n0;v = v*m0
    #meshgrid = kornia.create_meshgrid(m, n, normalized_coordinates=False) # Python可以选择是否归一化
    meshgrid = kornia.create_meshgrid(m, n, normalized_coordinates=True) # 与 VIFB 一致，归一化然后放缩，但精度不够，没有替代方案
    u = (meshgrid[0, :, :, 0]).to(torch.float64) * n0
    v = (meshgrid[0, :, :, 1]).to(torch.float64) * m0

    # Dog in Frequency
    r = torch.sqrt(u**2 + v**2)
    Sd_freq_domain = torch.exp(-(r / f0)**2) - a * torch.exp(-(r / f1)**2)
    if mode == 'frequency': 
        return Sd_freq_domain
    elif mode == 'spatial':
        Sd_time_domain = torch.fft.ifft2(Sd_freq_domain)
        Sd_time_domain /= torch.max(torch.abs(Sd_time_domain))
        return Sd_time_domain.real

def contrast_sensitivity_filtering_freq(im, mode='spatial'):
    # 计算 Sd 用于滤波
    Sd = contrast_sensitivity_filtering_Sd(im.size(),mode)
    #print(Sd.shape)
    if mode == 'frequency': # VIFB 的方法, 但是会导致梯度消失
        # 进行二维傅里叶变换
        im_fft = torch.fft.fft2(im)
        # if im.is_leaf == False:
        #     im_fft.retain_grad()
        #     im_fft.real.mean().backward()
        #     print (im_fft.grad)
        #     raise
        # fftshift 操作
        im_fft_shifted = torch.roll(im_fft, shifts=(im.shape[2]//2, im.shape[3]//2), dims=(2, 3))
        # if im_fft_shifted.is_leaf == False:
        #     im_fft_shifted.retain_grad()
        #     im_fft_shifted.real.mean().backward()
        #     print (im_fft_shifted.grad)
        #     raise
        # 点乘 Sd
        im_filtered_shifted = im_fft_shifted * Sd
        # if im_filtered_shifted.is_leaf == False:
        #     im_filtered_shifted.retain_grad()
        #     im_filtered_shifted.real.mean().backward()
        #     print (im_filtered_shifted.grad)
        #     raise
        # ifftshift 操作
        im_filtered = torch.roll(im_filtered_shifted, shifts=(-im.shape[2]//2, -im.shape[3]//2), dims=(2, 3))
        # if im_filtered.is_leaf == False:
        #     im_filtered.retain_grad()
        #     im_filtered.real.mean().backward()
        #     print (im_filtered.grad)
        #     raise
        # 逆二维傅里叶变换
        im = torch.fft.ifft2(im_filtered)
        # if im.is_leaf == False:
        #     im.retain_grad()
        #     im.real.mean().backward()
        #     print (im.grad)
        #     raise
        return im
    elif mode == 'spatial':
        # 使用时域卷积操作进行滤波
        return F.conv2d(im, Sd.unsqueeze(0).unsqueeze(0), padding=Sd.size(0)//2)
    else:
        raise Exception("mode should only be `spatial` or `frequency`")

def q_cb(imgA, imgB, imgF, border_type='constant', mode='spatial'):
    # mode = 'spatial', mode = 'frequency'
    # Normalize
    #imgF.requires_grad = True

    # return torch.mean(imgA-imgF)
    imgA = _normalize(imgA)
    imgB = _normalize(imgB)
    imgF = _normalize(imgF)
    # return torch.mean(imgA-imgF)

    # Contrast sensitivity filtering with DoG --- Get Sd
    # Sd 被用于计算图像的局部对比度，以评估融合图像的质量。
    # Sd = contrast_sensitivity_filtering_Sd(imgA.size())
    #print('1. Sd:',torch.mean(Sd))

    # Contrast sensitivity filtering with DoG --- Frequency Domain
    fused1 = contrast_sensitivity_filtering_freq(imgA,mode)
    fused2 = contrast_sensitivity_filtering_freq(imgB,mode)
    ffused = contrast_sensitivity_filtering_freq(imgF,mode)
    # if ffused.is_leaf == False:
    #     ffused.retain_grad()
    #     ffused.real.mean().backward()
    #     print (ffused.grad)
    #     raise
    # return torch.mean(ffused.real)
    # fused1 = contrast_sensitivity_filtering_freq(F.leaky_relu(imgA), Sd)
    # fused2 = contrast_sensitivity_filtering_freq(F.leaky_relu(imgB), Sd)
    # ffused = contrast_sensitivity_filtering_freq(F.leaky_relu(imgF), Sd)

    #print(torch.mean(fused1))
    #print(torch.mean(fused2))
    #print(torch.mean(ffused))

    # local contrast computation
    # G1, G2
    #print(torch.mean(G1),G1.shape)
    #print(torch.mean(G2),G2.shape)

    # filtering in frequency domain
    def filtering_in_frequency_domain(im):
        k = 1;h = 1;p = 3;q = 2;Z = 0.0001
        buff1 = kornia.filters.filter2d(im,G1.unsqueeze(0), border_type=border_type)
        buff2 = kornia.filters.filter2d(im,G2.unsqueeze(0), border_type=border_type)
        C = torch.abs(buff1.squeeze() / buff2.squeeze() - 1)
        #print(torch.mean(buff1),torch.mean(buff2),torch.mean(C))
        return (k * (C**p)) / (h * (C**q) + Z)
    C1P = filtering_in_frequency_domain(fused1)
    C2P = filtering_in_frequency_domain(fused2)
    CfP = filtering_in_frequency_domain(ffused)

    # if CfP.is_leaf == False:
    #     CfP.retain_grad()
    #     CfP.real.mean().backward()
    #     print (CfP.grad)
    #     raise

    #print(C1P.shape)
    #print(torch.mean(C1P))
    #print(torch.mean(C2P))
    #print(torch.mean(CfP))

    # contrast preservation calculation
    mask = (C1P < CfP).double()
    Q1F = (C1P / CfP) * mask + (CfP / C1P) * (1 - mask)

    mask = (C2P < CfP).double()
    Q2F = (C2P / CfP) * mask + (CfP / C2P) * (1 - mask)

    # Saliency map generation
    ramda1 = (C1P**2) / (C1P**2 + C2P**2)
    ramda2 = (C2P**2) / (C1P**2 + C2P**2)

    # global quality map
    Q = ramda1 * Q1F + ramda2 * Q2F

    return torch.mean(Q)


def q_cb_loss(imgA, imgB, imgF):
  return -q_cb(imgA, imgB, imgF)


def main():
    torch.manual_seed(42)  # 设置随机种子

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img1_path = '../imgs/TNO/vis/9.bmp'
    img2_path = '../imgs/TNO/ir/9.bmp'
    fused_path = '../imgs/TNO/fuse/U2Fusion/9.bmp'

    transform = transforms.Compose(
      [
          transforms.ToTensor(),
      ]
    )

    img1 = TF.to_tensor(Image.open(img1_path)).double().unsqueeze(0).to(device)
    img2 = TF.to_tensor(Image.open(img2_path)).double().unsqueeze(0).to(device)
    fused = TF.to_tensor(Image.open(fused_path)).double().unsqueeze(0).to(device)
    # print (img1, img2, fused)

    print(q_cb(img1,img2,fused,mode='frequency'))
    print(q_cb(img1,img1,img1,mode='frequency'))
    print(q_cb(img1,img2,fused,mode='spatial'))
    print(q_cb(img1,img1,img1,mode='spatial'))

if __name__ == '__main__':
  main()

