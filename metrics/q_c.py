import torch
import kornia

###########################################################################################

__all__ = [
    'q_c',
    'q_c_approach_loss',
    'q_c_metric'
]

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

    print(f'q_c(ir):')
    print(f'q_c(vis):')
    print(f'q_c(fused):')

if __name__ == '__main__':
    main()
