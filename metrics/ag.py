import torch
import kornia

###########################################################################################

__all__ = [
    'ag',
    'ag_approach_loss',
    'ag_metric'
]

def ag(tensor, eps=1e-10): # 默认输入的是 0-1 的浮点数
    """
    Calculate the edge-aware gradient (AG) of a tensor.

    Args:
        tensor (torch.Tensor): Input tensor, assumed to be in the range [0, 1].
        eps (float, optional): A small value to avoid numerical instability. Default is 1e-10.

    Returns:
        torch.Tensor: The edge-aware gradient of the input tensor.
    """
    # 使用Sobel算子计算水平和垂直梯度
    _grad_x = kornia.filters.filter2d(tensor,torch.tensor([[-1,  1]], dtype=torch.float64).unsqueeze(0))
    _grad_y = kornia.filters.filter2d(tensor,torch.tensor([[-1],[1]], dtype=torch.float64).unsqueeze(0))

    # 对梯度进行平均以避免过度敏感性(与 Matlab 统一)
    grad_x = (torch.cat((_grad_x[:,:,:,0:1],_grad_x[:,:,:,:-1]),dim=-1)+torch.cat((_grad_x[:,:,:,:-1],_grad_x[:,:,:,-2:-1]),dim=-1))/2
    grad_y = (torch.cat((_grad_y[:,:,0:1,:],_grad_y[:,:,:-1,:]),dim=-2)+torch.cat((_grad_y[:,:,:-1,:],_grad_y[:,:,-2:-1,:]),dim=-2))/2

    # 计算梯度的平均幅度
    s = torch.sqrt((grad_x ** 2 + grad_y ** 2 + eps)/2)

    # 返回 AG 值
    return torch.sum(s) / ((tensor.shape[2] - 1) * (tensor.shape[3] - 1)) * 255.0 # 与 VIFB 统一，需要乘 255

# 两张图一样，平均梯度会相等
def ag_approach_loss(A, F):
    return torch.abs(ag(A)-ag(F))

# 与 VIFB 统一
def ag_metric(A, B, F):
    return ag(F)

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

    print(f'AG(ir):{ag(ir)}')
    print(f'AG(vis):{ag(vis)}')
    print(f'AG(fused):{ag(fused)}')

if __name__ == '__main__':
    main()
