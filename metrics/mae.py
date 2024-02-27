import torch
import torch.nn.functional as F

###########################################################################################

__all__ = [
    'mae',
    'mae_approach_loss',
    'mae_metric'
]

def mae(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values.

    Args:
        y_true (torch.Tensor): The true values tensor.
        y_pred (torch.Tensor): The predicted values tensor.
        eps (float, optional): A small value to avoid numerical instability. Default is 1e-10.

    Returns:
        torch.Tensor: The MAE between true and predicted values.
    """
    return torch.mean(torch.abs(y_true - y_pred))

mae_approach_loss = mae

def mae_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    w0 = w1 = 0.5
    return w0 * mae(A, F) + w1 * mae(B, F)

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

    print(f'MAE(ir,ir):{mae(ir,ir)}')
    print(f'MAE(ir,vis):{mae(ir,vis)}')
    print(f'MAE(ir,fused):{mae(ir,fused)}')

if __name__ == '__main__':
    main()
