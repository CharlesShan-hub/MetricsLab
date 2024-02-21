import torch
import numpy as np

###########################################################################################

__all__ = [
    'ergas','ergas_numpy',
    'ergas_approach_loss',
    'ergas_metric'
]

def ergas(A: torch.Tensor, F: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Calculate the Error Relative Global Accuracy (ERGAS) metric between images A and F.

    Args:
    - A (torch.Tensor): The original image.
    - F (torch.Tensor): The fused image.
    - eps (float): A small value to avoid division by zero. Default is 1e-10.

    Returns:
    - torch.Tensor: The computed ERGAS metric.
    """
    # Get the image dimensions
    _, _, height, width = A.shape

    # Compute the mean squared error (MSE) and
    # root mean squared error (RMSE) for each band
    mse = torch.mean((A - F) ** 2)
    rmse = torch.sqrt(mse)

    # Compute the average brightness for each band
    average_brightness = torch.mean(A)

    # Define a constant parameter
    L = 256  # Assuming 256 grayscale levels

   # Compute ERGAS
    ergas = (100 / L) * torch.sqrt(1 / mse) * (rmse / average_brightness) ** 2
    return torch.sqrt(ergas)

def ergas_numpy(reference_image: np.ndarray, processed_image: np.ndarray) -> float:
    """
    Calculate the Error Relative Global Accuracy (ERGAS) metric between reference and processed images.
    https://blog.csdn.net/qq_49729636/article/details/134502721

    Args:
    - reference_image (numpy.ndarray): The reference image.
    - processed_image (numpy.ndarray): The processed/fused image.

    Returns:
    - float: The computed ERGAS metric.
    """
    # 获取图像的尺寸
    if len(reference_image.shape) == 2:
        height, width = reference_image.shape
        num_bands = 1
        reference_image=reference_image.reshape(height, width,num_bands)
        processed_image=processed_image.reshape(height, width,num_bands)
    else:
        height, width, num_bands = reference_image.shape

    # 初始化变量用于计算各个波段的MSE和RMSE
    mse_values = []
    rmse_values = []

    for band in range(num_bands):
        # 计算MSE（均方误差）
        mse = np.mean((reference_image[:, :, band] - processed_image[:, :, band]) ** 2)
        mse_values.append(mse)

        # 计算RMSE（均方根误差）
        rmse = np.sqrt(mse)
        rmse_values.append(rmse)

    # 计算每个波段的平均亮度
    average_brightness = [np.mean(reference_image[:, :, band]) for band in range(num_bands)]

    # 定义常数参数
    N = num_bands
    L = 256  # 假设灰度级数为256

    # 计算ERGAS
    ergas_values = []
    for mse, rmse, Y in zip(mse_values, rmse_values, average_brightness):
        ergas_values.append((100 / L) * np.sqrt(1 / mse) * (rmse / Y) ** 2)
    ergas = np.sqrt((1 / N) * np.sum(ergas_values))

    return ergas

def ergas_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return -ergas(A,F)

def ergas_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return ergas(A*255,F*255)+ergas(B*255,F*255)

###########################################################################################

def main():
    from torchvision import transforms
    from torchvision.transforms.functional import to_tensor
    from PIL import Image

    torch.manual_seed(42)

    transform = transforms.Compose([transforms.ToTensor()])

    vis_image = Image.open('../imgs/TNO/vis/9.bmp')
    vis_array = np.array(vis_image).astype(np.int32)
    vis_tensor = to_tensor(vis_image).unsqueeze(0)
    ir_image = np.array(Image.open('../imgs/TNO/ir/9.bmp'))
    ir_array = np.array(ir_image).astype(np.int32)
    ir_tensor = to_tensor(ir_image).unsqueeze(0)
    fused_image = np.array(Image.open('../imgs/TNO/fuse/U2Fusion/9.bmp'))
    fused_array = np.array(fused_image).astype(np.int32)
    fused_tensor = to_tensor(fused_image).unsqueeze(0)

    print(f'ergas(ir,fused) by numpy:{ergas_numpy(ir_array,fused_array)}')
    print(f'ergas(ir,fused) by self: {ergas(ir_tensor*255,fused_tensor*255)}')
    print(f'ergas(ir,ir) by self: {ergas(ir_tensor*255,ir_tensor*255)}')
    print(f'ergas(ir,vis) by self: {ergas(ir_tensor*255,vis_tensor*255)}')


if __name__ == '__main__':
    main()
