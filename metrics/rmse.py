import torch
import torch.nn.functional as F

###########################################################################################

__all__ = [
    'rmse',
    'rmse_approach_loss',
    'rmse_metric'
]

def rmse(y_true, y_pred, eps=1e-10):
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (torch.Tensor): The true values tensor.
        y_pred (torch.Tensor): The predicted values tensor.
        eps (float, optional): A small value to avoid numerical instability. Default is 1e-10.

    Returns:
        torch.Tensor: The RMSE between true and predicted values.
    """
    mse_loss = torch.mean((y_true - y_pred)**2)
    rmse_loss = torch.sqrt(mse_loss + eps)
    return rmse_loss

rmse_approach_loss = rmse

# 与 VIFB 统一
def rmse_metric(A, B, F):
    w0 = w1 = 0.5
    return w0 * rmse(A, F) + w1 * rmse(B, F)

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

    print(f'RMSE(ir,ir):{rmse(ir,ir)}')
    print(f'RMSE(ir,vis):{rmse(ir,vis)}')
    print(f'RMSE(ir,fused):{rmse(ir,fused)}')

if __name__ == '__main__':
    main()
