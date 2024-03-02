import torch
import kornia

###########################################################################################

__all__ = [
    'q_h',
    'q_h_approach_loss',
    'q_h_metric'
]

def q_h(A,B,F,window_size = 7):
    def _ssim(X,Y):
        C1 = (0.01*255)**2
        C2 = (0.03*255)**2
        kernel = kornia.filters.get_gaussian_kernel1d(window_size, 1.5, device=X.device, dtype=X.dtype)
        muA = kornia.filters.filter2d_separable(X, kernel, kernel, padding="valid")
        muB = kornia.filters.filter2d_separable(Y, kernel, kernel, padding="valid")
        sAA = kornia.filters.filter2d_separable(X**2, kernel, kernel, padding="valid") - muA**2
        sBB = kornia.filters.filter2d_separable(Y**2, kernel, kernel, padding="valid") - muB**2
        sAB = kornia.filters.filter2d_separable(X*Y, kernel, kernel, padding="valid") - muA*muB

        return  ((2*muA*muB + C1)*(2*sAB + C2)) / ((muA**2 + muB**2 + C1)*(sAA + sBB + C2));



def q_h_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return 1 - q_h(A,F)

def q_h_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    w0 = w1 = 0.5
    return w0 * q(A, F) + w1 * q(B, F)

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

    print(f'Q_H(vis,ir,fused):{q_h(vis,ir,fused)}')
    print(f'Q_H(vis,vis,vis):{q_h(vis,vis,vis)}')
    print(f'Q_H(vis,vis,fused):{q_h(vis,vis,fused)}')
    print(f'WFQ_HQI(vis,vis,ir):{q_h(vis,vis,ir)}')

if __name__ == '__main__':
    main()
