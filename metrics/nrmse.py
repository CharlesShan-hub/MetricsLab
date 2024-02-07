import torch

###########################################################################################

__all__ = [
    'nrmse',
    'nrmse_approach_loss',
    'nrmse_metric'
]

def nrmse(y_true, y_pred, eps=1e-10):
    # https://blog.csdn.net/weixin_43465015/article/details/105524728
    mse_loss = torch.mean((y_true - y_pred)**2)
    rmse_loss = torch.sqrt(mse_loss + eps)
    nrmse_loss = rmse_loss / (torch.max(y_pred)-torch.min(y_pred)+eps)
    return nrmse_loss

nrmse_approach_loss = nrmse

def nrmse_metric(A, B, F):
    w0 = w1 = 0.5
    return w0 * nrmse(A, F) + w1 * nrmse(B, F)

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

    print(f'NRMSE(ir,ir):{nrmse(ir,ir)}')
    print(f'NRMSE(ir,vis):{nrmse(ir,vis)}')
    print(f'NRMSE(ir,fused):{nrmse(ir,fused)}')

if __name__ == '__main__':
    main()
