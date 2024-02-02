import pytorch_msssim
import torch

###########################################################################################

__all__ = [
    'ms_ssim',
    'ms_ssim_approach_loss',
    'ms_ssim_metric'
]

# https://github.com/VainF/pytorch-msssim
def ms_ssim(X, Y, data_range=1, size_average=False):
    return pytorch_msssim.ms_ssim(X,Y,data_range,size_average)

# https://github.com/VainF/pytorch-msssim
def ms_ssim_approach_loss(X, Y, data_range=1, size_average=False):
    return 1 - ms_ssim(X,Y,data_range,size_average)

def ms_ssim_metric(A, B, F):
    w0 = w1 = 0.5
    return torch.mean(w0 * ms_ssim(A, F) + w1 * ms_ssim(B ,F))

###########################################################################################

def main():
    from torchvision import transforms
    from torchvision.transforms.functional import to_tensor
    from PIL import Image

    torch.manual_seed(42)

    transform = transforms.Compose([transforms.ToTensor()])

    vis = to_tensor(Image.open('../imgs/TNO/vis/9.bmp')).unsqueeze(0)
    ir = to_tensor(Image.open('../imgs/TNO/ir/9.bmp')).unsqueeze(0)
    fused = to_tensor(Image.open('../imgs/TNO/fuse/U2Fusion/9.bmp')).unsqueeze(0)

    print(f'MSSIM(ir,ir):{torch.mean(ms_ssim(ir,ir))}')
    print(f'MSSIM(ir,fused):{torch.mean(ms_ssim(ir,fused))}')
    print(f'MSSIM(vis,fused):{torch.mean(ms_ssim(vis,fused))}')

if __name__ == '__main__':
    main()
