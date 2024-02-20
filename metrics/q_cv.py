import torch
import kornia

###########################################################################################

__all__ = [
    'q_cv',
    'q_cv_approach_loss',
    'q_cv_metric'
]

def _normalize(data):
    max_value = torch.max(data)
    min_value = torch.min(data)
    if max_value == 0 and min_value == 0: return data
    else: newdata = (data - min_value) / (max_value - min_value)
    return newdata * 255
    #return torch.round(newdata * 255) # <- 这一步会导致梯度消失

def _freq_meshgrid(size, d=1.0):
    """生成频率坐标"""
    def _line(n, d=1.0):
        val = 1.0 / (n * d)
        results = torch.arange(0, n).to(torch.float)
        p2 = (n + 1) // 2
        results[p2:] -= n
        results *= val
        shift = p2 if n % 2 == 0 else p2 - 1
        return torch.roll(results, shift, dims=0)  # 将零频率移到中心
    _, _, m, n = size
    return torch.meshgrid(_line(m,d)*2, _line(n,d)*2, indexing='ij')

def q_cv(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor, window_size: int = 16,
          border_type: str = 'constant', normalize: bool = True, eps: float = 1e-10) -> torch.Tensor:
    """
    Calculate the Q_CV (Quality Assessment for image Corner and Vertices) metric for image fusion.

    Args:
        A (torch.Tensor): The first input image tensor.
        B (torch.Tensor): The second input image tensor.
        F (torch.Tensor): The fused image tensor.
        window_size (int, optional): The size of the window for local region saliency calculation. Default is 16.
        border_type (str, optional): Type of border extension. Default is 'constant'.
        normalize (bool, optional): Whether to normalize input images to the range [0, 1]. Default is True.
        eps (float, optional): A small value to avoid numerical instability. Default is 1e-10.

    Returns:
        torch.Tensor: The Q_CV metric value.
    """
    # Step 0: Params and Normalize
    alpha_c=1
    alpha_s=0.685
    f_c=97.3227
    f_s=12.1653
    alpha=5 #alpha = 1, 2, 3, 4, 5, 10, 15. This value is adjustable.;-)
    if normalize:
        [A, B, F] = [_normalize(I) for I in [A, B, F]]
    else:
        [A, B, F] = [I*255 for I in [A, B, F]]

    # Step 1: extract edge information
    def graident(A):
        grad_x = kornia.filters.filter2d(A,torch.tensor([[[-1,  0, 1],[-2,0,2],[-1,0,1]]]),border_type=border_type)
        grad_y = kornia.filters.filter2d(A,torch.tensor([[[-1, -2,-1],[ 0,0,0],[ 1,2,1]]]),border_type=border_type)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + eps)
    [GA, GB, GF] = [graident(I) for I in [A, B, F]]

    # Step 2: padding and calculate the local region saliency
    # 经过 Padding 和 VIFB 结果一致
    origin_H, origin_L = (A.shape[-2] // window_size, A.shape[-1] // window_size)
    pad_H = (window_size - origin_H % window_size) % window_size
    pad_L = (window_size - origin_L % window_size) % window_size
    GA = torch.nn.functional.pad(GA, (0, pad_H, 0, pad_L), mode='constant', value=0)
    GB = torch.nn.functional.pad(GB, (0, pad_H, 0, pad_L), mode='constant', value=0)
    H, L = (GA.shape[-2] // window_size, GA.shape[-1] // window_size)
    RA = torch.zeros(H, L); RB = torch.zeros(H, L)
    for i in range(H):
        for j in range(L):
            block = GA[:,:, i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]
            RA[i, j] = torch.sum(torch.pow(block, alpha))
            block = GB[:,:, i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]
            RB[i, j] = torch.sum(torch.pow(block, alpha))

    # Step 3. similarity measurement
    DA = A - F
    DB = B - F
    u, v = _freq_meshgrid(A.shape)
    u = u * (A.shape[-2]/8) # VIFB里边写的 8
    v = v * (A.shape[-1]/8) # VIFB里边写的 8
    r = torch.sqrt(u**2 + v**2)
    theta_m = 2.6 * (0.0192 + 0.144 * r) * torch.exp(-torch.pow(0.144 * r, 1.1)) # Mannos-Skarison's filter
    def filter(D):
        FD = torch.fft.fft2(D)
        FDS = torch.roll(FD, shifts=(FD.shape[2]//2, FD.shape[3]//2), dims=(2, 3))
        FDS = FDS * theta_m
        FD = torch.roll(FDS, shifts=(-FDS.shape[2]//2, -FDS.shape[3]//2), dims=(2, 3))
        return torch.fft.ifft2(FD)
    DFA = torch.nn.functional.pad(filter(DA).real, (0, pad_H, 0, pad_L), mode='constant', value=0)
    DFB = torch.nn.functional.pad(filter(DB).real, (0, pad_H, 0, pad_L), mode='constant', value=0)
    MDA = torch.zeros(H, L); MDB = torch.zeros(H, L)
    for i in range(H):
        for j in range(L):
            block = DFA[:,:, i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]
            MDA[i, j] = torch.mean(torch.pow(block, 2))
            block = DFB[:,:, i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]
            MDB[i, j] = torch.mean(torch.pow(block, 2))
    return torch.sum(RA*MDA+RB*MDB) / torch.sum(RA+RB)

def q_cv_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return q_cv(A, A, F, window_size=16, border_type='constant', normalize=True, eps=1e-10)

def q_cv_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return q_cv(A, B, F, window_size=16, border_type='constant', normalize=True, eps=1e-10)

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

    print(f'QCV(ir,vis,fused):{q_cv(vis,ir,fused)}')
    print(f'QCV(vis,vis,vis):{q_cv(vis,vis,vis)}')
    print(f'QCV(vis,vis,fused):{q_cv(vis,vis,fused)}')
    print(f'QCV(vis,vis,ir):{q_cv(vis,vis,ir)}')

if __name__ == '__main__':
    main()
