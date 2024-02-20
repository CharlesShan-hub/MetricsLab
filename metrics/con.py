import torch
import kornia

###########################################################################################

__all__ = [
    'con',
    'con_approach_loss',
    'con_metric'
]

def con(A: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Contrast (CON) metric of an image.
    https://blog.csdn.net/zsc201825/article/details/89645190

    Args:
        A (torch.Tensor): The input image tensor.

    Returns:
        torch.Tensor: The calculated CON value.
    """
    # padding
    A = torch.nn.functional.pad(A*255, (1, 1, 1, 1), mode='replicate')
    # con
    res1 = kornia.filters.filter2d(A,torch.tensor([[[0,1,0],[0,-1,0],[0,0,0]]]),padding='valid')
    res2 = kornia.filters.filter2d(A,torch.tensor([[[0,0,0],[1,-1,0],[0,0,0]]]),padding='valid')
    res3 = kornia.filters.filter2d(A,torch.tensor([[[0,0,0],[0,-1,1],[0,0,0]]]),padding='valid')
    res4 = kornia.filters.filter2d(A,torch.tensor([[[0,0,0],[0,-1,0],[0,1,0]]]),padding='valid')

    res1 = torch.sum(torch.abs(res1) ** 2)
    res2 = torch.sum(torch.abs(res2) ** 2)
    res3 = torch.sum(torch.abs(res3) ** 2)
    res4 = torch.sum(torch.abs(res4) ** 2)
    _,_,M,N = A.shape
    M-=2
    N-=2
    return (res1+res2+res3+res4)/(4*M*N - 2*M - 2*N)

def con_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return torch.abs(con(A)-con(F))

def con_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return con(F)

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
    toy = torch.tensor([[[[1,3,9,9],[2,1,3,7],[3,6,0,6],[6,8,2,0]]]])/255.0

    print(f'CON(toy):{con(toy)}') # should be 13.333
    print(f'CON(ir):{con(ir)}')
    print(f'CON(vis):{con(vis)}')
    print(f'CON(fused):{con(fused)}')

if __name__ == '__main__':
    main()
