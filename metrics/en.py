import torch
import kornia
import numpy as np
import matplotlib.pyplot as plt

def entropy(grey_tensor,bandwidth=0.1,eps=1e-10):
    grey_tensor = grey_tensor.view(1, -1) * 255
    bins = torch.linspace(0, 255, 256).to(grey_tensor.device)
    histogram = kornia.enhance.histogram(grey_tensor, bins=bins, bandwidth=torch.tensor(bandwidth))
    image_entropy = -torch.sum(histogram * torch.log2(histogram + eps))
    return image_entropy

def entropy_loss(grey_tensor,grey_scale=256):
    return np.log2(grey_scale)-entropy(grey_tensor)

def en_metric(imgA,imgB,imgF):
    return entropy(imgF)

def demo1():
    # 计算理论最大值
    print("理论最大值(log2 n)：", np.log2(16 * 16))

    # 生成完全完全覆盖所有像素的张量
    full_tensor = torch.arange(256).view(1, 1, 16, 16)
    full_tensor_entropy = entropy(full_tensor)
    print("均匀张量的信息熵：", full_tensor_entropy)

    # 生成随机张量
    random_tensor = torch.randint(0, 256, size=(1, 1, 16, 16), dtype=torch.uint8)
    random_tensor_entropy = entropy(random_tensor)
    print("随机张量的信息熵：", random_tensor_entropy)

    # 生成纯色张量
    white_tensor = torch.ones(1, 1, 16, 16) * 255
    white_tensor_entropy = entropy(white_tensor)
    print("白色张量的信息熵：", white_tensor_entropy)

    grey_tensor = torch.ones(1, 1, 16, 16) * 127
    grey_tensor_entropy = entropy(grey_tensor)
    print("灰色张量的信息熵：", grey_tensor_entropy)

    black_tensor = torch.ones(1, 1, 16, 16) * 0
    black_tensor_entropy = entropy(black_tensor)
    print("黑色张量的信息熵：", black_tensor_entropy)

    # 绘制图像
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))

    # 均匀张量
    axs[0].imshow(full_tensor.view(16, 16).numpy(), cmap='gray', vmin=0, vmax=255)
    axs[0].set_title(f'Uniform\nEntropy: {full_tensor_entropy:.2f}')

    # 随机张量
    axs[1].imshow(random_tensor.view(16, 16).numpy(), cmap='gray', vmin=0, vmax=255)
    axs[1].set_title(f'Random\nEntropy: {random_tensor_entropy:.2f}')

    # 灰色张量
    axs[2].imshow(grey_tensor.view(16, 16).numpy(), cmap='gray', vmin=0, vmax=255)
    axs[2].set_title(f'Grey\nEntropy: {grey_tensor_entropy:.2f}')

    # 白色张量
    axs[3].imshow(white_tensor.view(16, 16).numpy(), cmap='gray', vmin=0, vmax=255)
    axs[3].set_title(f'White\nEntropy: {white_tensor_entropy:.2f}')

    # 黑色张量
    axs[4].imshow(black_tensor.view(16, 16).numpy(), cmap='gray', vmin=0, vmax=255)
    axs[4].set_title(f'Black\nEntropy: {black_tensor_entropy:.2f}')

    plt.show()

def demo2():
    # 计算理论最大值
    print("理论最大值(log2 n)：", np.log2(16 * 16))

    # 生成完全完全覆盖所有像素的张量
    full_tensor = torch.arange(256*16).view(1, 1, 64, 64) / 16
    full_tensor_entropy = entropy(full_tensor)
    print("均匀张量的信息熵：", full_tensor_entropy)

    # 生成随机张量
    random_tensor = torch.rand(1, 1, 64, 64) * 255
    random_tensor_entropy = entropy(random_tensor)
    print("随机张量的信息熵：", random_tensor_entropy)

    # 生成纯色张量
    white_tensor = torch.ones(1, 1, 64, 64) * 255
    white_tensor_entropy = entropy(white_tensor)
    print("白色张量的信息熵：", white_tensor_entropy)

    grey_tensor = torch.ones(1, 1, 64, 64) * 127
    grey_tensor_entropy = entropy(grey_tensor)
    print("灰色张量的信息熵：", grey_tensor_entropy)

    black_tensor = torch.ones(1, 1, 64, 64) * 0
    black_tensor_entropy = entropy(black_tensor)
    print("黑色张量的信息熵：", black_tensor_entropy)

    half_tensor = torch.cat((torch.zeros(1, 1, 64, 32), torch.ones(1, 1, 64, 32) * 255), dim=3)
    half_tensor_entropy = entropy(half_tensor)
    print("黑白张量的信息熵：", half_tensor_entropy)

    # 绘制图像
    fig, axs = plt.subplots(1, 6, figsize=(24, 4))

    # 均匀张量
    axs[0].imshow(full_tensor.view(64, 64).numpy(), cmap='gray', vmin=0, vmax=255)
    axs[0].set_title(f'Uniform\nEntropy: {full_tensor_entropy:.2f}')

    # 随机张量
    axs[1].imshow(random_tensor.view(64, 64).numpy(), cmap='gray', vmin=0, vmax=255)
    axs[1].set_title(f'Random\nEntropy: {random_tensor_entropy:.2f}')

    # 灰色张量
    axs[2].imshow(half_tensor.view(64, 64).numpy(), cmap='gray', vmin=0, vmax=255)
    axs[2].set_title(f'Half\nEntropy: {half_tensor_entropy:.2f}')

    # 灰色张量
    axs[3].imshow(grey_tensor.view(64, 64).numpy(), cmap='gray', vmin=0, vmax=255)
    axs[3].set_title(f'Grey\nEntropy: {grey_tensor_entropy:.2f}')

    # 白色张量
    axs[4].imshow(white_tensor.view(64, 64).numpy(), cmap='gray', vmin=0, vmax=255)
    axs[4].set_title(f'White\nEntropy: {white_tensor_entropy:.2f}')

    # 黑色张量
    axs[5].imshow(black_tensor.view(64, 64).numpy(), cmap='gray', vmin=0, vmax=255)
    axs[5].set_title(f'Black\nEntropy: {black_tensor_entropy:.2f}')

    plt.show()

def demo3():
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    # 读取图片
    image = cv2.imread('../imgs/TNO/fuse/ADF/9.bmp', cv2.IMREAD_GRAYSCALE)

    # 转换为 tensor
    image_tensor = kornia.image_to_tensor(image).float()

    # 使用 numpy 统计直方图
    hist_np, bin_edges_np = np.histogram(image_tensor.numpy().flatten(), bins=256, range=[0, 256], density=True)

    # 使用 kornia 核密度直方图
    image_tensor = image_tensor.view(1, -1)
    bins = torch.linspace(0, 255, 256).to(image_tensor.device)
    histogram_result = kornia.enhance.histogram(image_tensor, bins=bins, bandwidth=torch.tensor(1))

    # 作图
    # plt.figure(figsize=(10, 6))

    plt.subplot(2,3,1)
    print(hist_np)
    plt.title(f'Numpy, EN={-np.sum(hist_np * np.log2(hist_np + 1e-10))}')
    plt.plot(bin_edges_np[:-1], hist_np, color='blue', label='Numpy Histogram')
    plt.subplot(2,3,2)
    plt.title(f'Kornia (sigma = 1), EN={-torch.sum(histogram_result * torch.log2(histogram_result + 1e-10))}')
    plt.plot(histogram_result.squeeze().detach().numpy(), color='orange', label='Kornia Histogram')
    plt.subplot(2,3,3)
    plt.plot(bin_edges_np[:-1], hist_np, color='blue', label='Numpy Histogram')
    plt.plot(histogram_result.squeeze().detach().numpy(), color='orange', label='Kornia Histogram')
    plt.title('Overlapping')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Normalized Frequency')
    plt.subplot(2,3,4)
    histogram_result = kornia.enhance.histogram(image_tensor, bins=bins, bandwidth=torch.tensor(0.1))
    plt.title(f'Kornia (sigma = 0.1), EN={-torch.sum(histogram_result * torch.log2(histogram_result + 1e-10))}')
    plt.plot(histogram_result.squeeze().detach().numpy(), color='orange', label='Kornia Histogram')
    plt.subplot(2,3,5)
    histogram_result = kornia.enhance.histogram(image_tensor, bins=bins, bandwidth=torch.tensor(0.4))
    plt.title(f'Kornia (sigma = 0.4), EN={-torch.sum(histogram_result * torch.log2(histogram_result + 1e-10))}')
    plt.plot(histogram_result.squeeze().detach().numpy(), color='orange', label='Kornia Histogram')
    plt.subplot(2,3,6)
    histogram_result = kornia.enhance.histogram(image_tensor, bins=bins, bandwidth=torch.tensor(2))
    plt.title(f'Kornia (sigma = 2), EN={-torch.sum(histogram_result * torch.log2(histogram_result + 1e-10))}')
    plt.plot(histogram_result.squeeze().detach().numpy(), color='orange', label='Kornia Histogram')
    plt.legend()

    # 展示图像
    plt.show()

def main():
    demo1()
    #demo2()
    # demo3()

if __name__ == '__main__':
    main()
