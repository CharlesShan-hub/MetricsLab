import torch
import kornia

###########################################################################################

__all__ = [
    'ag',
    'ag_approach_loss',
    'ag_metric'
]

def ag(tensor, eps=1e-10): # 默认输入的是 0-1 的浮点数
    _grad_x = kornia.filters.filter2d(tensor,torch.tensor([[-1,  1]], dtype=torch.float64).unsqueeze(0))
    _grad_y = kornia.filters.filter2d(tensor,torch.tensor([[-1],[1]], dtype=torch.float64).unsqueeze(0))
    grad_x = (torch.cat((_grad_x[:,:,:,0:1],_grad_x[:,:,:,:-1]),dim=-1)+torch.cat((_grad_x[:,:,:,:-1],_grad_x[:,:,:,-2:-1]),dim=-1))/2
    grad_y = (torch.cat((_grad_y[:,:,0:1,:],_grad_y[:,:,:-1,:]),dim=-2)+torch.cat((_grad_y[:,:,:-1,:],_grad_y[:,:,-2:-1,:]),dim=-2))/2
    s = torch.sqrt((grad_x ** 2 + grad_y ** 2 + eps)/2)
    return torch.sum(s) / ((tensor.shape[2] - 1) * (tensor.shape[3] - 1))# * 255.0 # 与 VIFB 统一，需要乘 255

def average_gradient_loss(tensor):
    return -ag(tensor)

def ag_metric(A, B, F):
    return ag(F)

###########################################################################################

def main():
    from PIL import Image
    from torchvision import transforms
    import torchvision.transforms.functional as TF

    #tensor = TF.to_tensor(Image.open('../imgs/TNO/fuse/U2Fusion/9.bmp')).unsqueeze(0)
    #tensor = TF.to_tensor(Image.open('../imgs/TNO/vis/9.bmp')).unsqueeze(0)
    tensor = TF.to_tensor(Image.open('../imgs/TNO/ir/9.bmp')).unsqueeze(0)
    tensor = torch.clamp(torch.mul(tensor, 255), 0, 255).to(torch.float64)

    tensor = ag(tensor)
    print(tensor)
if __name__ == '__main__':
    main()
