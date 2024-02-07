import torch
import kornia

###########################################################################################

__all__ = [
    'vif',
    'vif_approach_loss',
    'vif_metric'
]

def vif():
    pass

def vif_approach_loss():
    pass

def vif_metric():
    pass

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

    print(f'VIF(ir):')
    print(f'VIF(vis):')
    print(f'VIF(fused):')

if __name__ == '__main__':
    main()
