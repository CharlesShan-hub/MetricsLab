import torch
import kornia

###########################################################################################

__all__ = [
    'q_y',
    'q_y_approach_loss',
    'q_y_metric'
]

def q_y(A, B, F, window_size=7, eps=1e-10):
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
        return (ssim_map,sAA,sBB)

    (ssimAB, SAA, SBB) = ssim_yang(A*255, B*255)
    (ssimAF, _, _) = ssim_yang(A*255, F*255)
    (ssimBF, _, _) = ssim_yang(B*255, F*255)

    ramda=SAA/(SAA+SBB+eps)

    Q1 = (ramda*ssimAF + (1-ramda)*ssimBF)[(ssimAB>=0.75) * ((SAA+SBB)>eps)]
    Q2 = torch.max(ssimAF,ssimBF)[(ssimAB<0.75) * ((SAA+SBB)>eps)]

    return (torch.sum(Q1)+torch.sum(Q2)) / (Q1.shape[0]+Q2.shape[0]) # 为了和 MEFB 统一，改变了平均的方式

def q_y_approach_loss():
    pass

def q_y_metric(A, B, F):
    return q_y(A, B, F, window_size=7, eps=1e-10)

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

    print(f'QY:{q_y_metric(vis, ir, fused)}')

if __name__ == '__main__':
    main()
