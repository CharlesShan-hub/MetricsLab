import torch
import kornia

###########################################################################################

__all__ = [
    'q_c',
    'q_c_approach_loss',
    'q_c_metric'
]

def q_c(A, B, F, window_size=7, eps=1e-10):
    def ssim_yang(A,B): # SSIM_Yang
        C1 = 2e-16
        C2 = 2e-16
        kernel = kornia.filters.get_gaussian_kernel1d(window_size, 1.5, device=A.device, dtype=A.dtype)
        muA = kornia.filters.filter2d_separable(A, kernel, kernel, padding="valid")
        muB = kornia.filters.filter2d_separable(B, kernel, kernel, padding="valid")
        sAA = kornia.filters.filter2d_separable(A**2, kernel, kernel, padding="valid") - muA**2
        sBB = kornia.filters.filter2d_separable(B**2, kernel, kernel, padding="valid") - muB**2
        sAB = kornia.filters.filter2d_separable(A*B, kernel, kernel, padding="valid") - muA*muB
        ssim_map = ((2*muA*muB + C1)*(2*sAB + C2)) / ((muA**2 + muB**2 + C1)*(sAA + sBB + C2)+eps)
        return (ssim_map,sAB)

    (ssimAF, SAF) = ssim_yang(A*255, F*255)
    (ssimBF, SBF) = ssim_yang(B*255, F*255)
    ssimABF = SAF / (SAF+SBF+eps)
    Q_C = ssimABF*ssimAF + (1-ssimABF)*ssimBF
    Q_C[ssimABF>1] = 1
    Q_C[ssimABF<0] = 0
    return torch.mean(Q_C)

def q_c_approach_loss(A, F):
    pass

def q_c_metric(A, B, F):
    return q_c(A, B, F, window_size=7)

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

    print(f'q_c_metric:{q_c(vis, ir, fused)}')

if __name__ == '__main__':
    main()
