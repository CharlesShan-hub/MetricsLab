import torch
import kornia

###########################################################################################

__all__ = [
    'ce',
    'ce_approach_loss',
    'ce_metric'
]

def ce(target, predict, bandwidth=0.1, eps=1e-10):
    """
    Calculate the cross-entropy between the target and predicted histograms.

    Args:
        target (torch.Tensor): The target image tensor.
        predict (torch.Tensor): The predicted image tensor.
        bandwidth (float, optional): Bandwidth for histogram smoothing. Default is 0.1.
        eps (float, optional): A small value to avoid numerical instability. Default is 1e-10.

    Returns:
        torch.Tensor: The cross-entropy between the histograms of the target and predicted images.
    """
    # 将预测值和目标值缩放到范围[0, 255]
    predict = predict.view(1, -1) * 255
    target = target.view(1, -1) * 255

    # 创建用于直方图计算的区间
    bins = torch.linspace(0, 255, 256).to(predict.device)

    # 计算目标和预测图像的直方图
    h1 = kornia.enhance.histogram(target, bins=bins, bandwidth=torch.tensor(bandwidth))
    h2 = kornia.enhance.histogram(predict, bins=bins, bandwidth=torch.tensor(bandwidth))

    # 创建一个掩码以排除直方图中小于eps的值 - 这里是与 VIFB 统一的重点
    mask = (h1 > eps)&( h2 > eps)

    # 计算交叉熵
    return torch.sum(h1[mask] * torch.log2(h1[mask]/(h2[mask])))

# 如果两幅图片一样 ce 为 0
def ce_approach_loss(A,F):
    return ce(A, F)

# 与 VIFB 统一
def ce_metric(A, B, F):
    w0 = w1 = 0.5
    return w0 * ce(A,F) + w1 * ce(B,F)

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
    rand = torch.randint(0, 255, size=fused.shape, dtype=torch.uint8)/255.0

    print("'Distance' with x and ir")
    print(f'CE(ir,ir):   {ce(ir,ir)}')
    print(f'CE(ir,vis):  {ce(ir,vis)}')
    print(f'CE(ir,fused):{ce(ir,fused)}')

    print("\nIf fused is fused | ir | vis  | average | rand")
    print(f'[Fused = fused]   CE(ir,fused)+CE(vis,fused):    {ce(ir,fused)+ce(vis,fused)}')
    print(f'[Fused = ir]      CE(ir,ir)+CE(vis,ir):          {ce(ir,ir)+ce(vis,ir)}')
    print(f'[Fused = vis]     CE(ir,vis)+CE(vis,vis):        {ce(ir,vis)+ce(vis,vis)}')
    print(f'[Fused = average] CE(ir,arverge)+CE(vis,arverge):{ce(ir,(vis+ir)/2)+ce(vis,(vis+ir)/2)}')
    print(f'[Fused = rand]    CE(ir,rand)+CE(vis,rand):      {ce(ir,rand)+ce(vis,rand)}')


if __name__ == '__main__':
    main()
