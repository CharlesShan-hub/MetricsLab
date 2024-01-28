import torch
import kornia
import numpy as np
import matplotlib.pyplot as plt

def edge_intensity(tensor):
    grad_x = kornia.filters.filter2d(tensor,torch.tensor([[-1,  0,  1],[-2,  0,  2],[-1,  0,  1]], dtype=torch.float32).unsqueeze(0))
    grad_y = kornia.filters.filter2d(tensor,torch.tensor([[-1, -2, -1],[ 0,  0,  0],[ 1,  2,  1]], dtype=torch.float32).unsqueeze(0))
    s = torch.sqrt((grad_x ** 2 + grad_y ** 2))
    return torch.mean(s)

def edge_intensity_loss(tensor):
    return -edge_intensity(tensor)

def main():
    from PIL import Image
    from torchvision import transforms
    import torchvision.transforms.functional as TF

    vis_tensor = TF.to_tensor(Image.open('../resources/imgs/vis/1.jpg')).unsqueeze(0)
    vis_tensor = torch.clamp(torch.mul(vis_tensor, 255), 0, 255).to(torch.uint8)

    tensor = edge_intensity(vis_tensor)
    image = tensor.squeeze().detach().numpy()
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    plt.imshow(image,cmap='grey')
    plt.show()
if __name__ == '__main__':
    main()