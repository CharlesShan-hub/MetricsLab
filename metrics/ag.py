import torch
import kornia

def average_gradient(tensor):
    grad_x = kornia.filters.filter2d(tensor,torch.tensor([[1,  -1]], dtype=torch.float32).unsqueeze(0))
    grad_y = kornia.filters.filter2d(tensor,torch.tensor([[1],[-1]], dtype=torch.float32).unsqueeze(0))
    s = torch.sqrt((grad_x ** 2 + grad_y ** 2))/4
    return torch.sum(s) / ((tensor.shape[2] - 1) * (tensor.shape[3] - 1))

def average_gradient_loss(tensor):
    return -average_gradient(tensor)

def main():
    from PIL import Image
    from torchvision import transforms
    import torchvision.transforms.functional as TF

    vis_tensor = TF.to_tensor(Image.open('../imgs/RoadScene/vis/1.jpg')).unsqueeze(0)
    vis_tensor = torch.clamp(torch.mul(vis_tensor, 255), 0, 255).to(torch.uint8)

    print(average_gradient(vis_tensor))
if __name__ == '__main__':
    main()