import torch
import kornia
# import matplotlib.pyplot as plt

###########################################################################################

__all__ = [
    'te',
    'te_approach_loss',
    'te_metric'
]

def te(image1: torch.Tensor, image2: torch.Tensor,
    q: float = 1.85, bandwidth: float = 0.1, eps: float = 1e-12,
    normalize: bool = False) -> torch.Tensor:
    """
    Calculate the Tsallis entropy (TE) between two input images.

    Args:
        image1 (torch.Tensor): The first input image tensor.
        image2 (torch.Tensor): The second input image tensor.
        q (float, optional): The Tsallis entropy parameter. Default is 1.85.
        bandwidth (float, optional): Bandwidth for histogram smoothing. Default is 0.1.
        eps (float, optional): A small value to avoid numerical instability. Default is 1e-10.
        normalize (bool, optional): Whether to normalize input images. Default is False.

    Returns:
        torch.Tensor: The Tsallis entropy between the two input images.
    """
    # 将图片拉平成一维向量,将一维张量转换为二维张量
    if normalize == True:
        x1 = ((image1-torch.min(image1))/(torch.max(image1) - torch.min(image1))).view(1,-1) * 255
        x2 = ((image2-torch.min(image2))/(torch.max(image2) - torch.min(image2))).view(1,-1) * 255
    else:
        x1 = image1.view(1,-1) * 255
        x2 = image2.view(1,-1) * 255

    # 定义直方图的 bins
    bins = torch.linspace(0, 255, 256).to(image1.device)

    # 计算二维直方图
    hist = kornia.enhance.histogram2d(x1, x2, bins, bandwidth=torch.tensor(bandwidth))

    # 计算边缘分布
    marginal_x = torch.sum(hist, dim=2)
    marginal_y = torch.sum(hist, dim=1)

    # plt.plot(marginal_x.squeeze().detach().numpy())
    # plt.show()
    # plt.plot(marginal_y.squeeze().detach().numpy())
    # plt.show()

    temp = marginal_x.unsqueeze(1) * marginal_y.unsqueeze(2) # 转置并广播
    mask = (temp > 10*eps)
    temp2 = (temp[mask]) ** (q-1)
    temp1 = hist[mask] ** q
    # print(torch.sum(temp1),torch.sum(temp2))
    result = torch.sum(hist[mask] ** q / (temp[mask]) ** (q-1))

    return (1-result)/(1-q)

# 两张图一样，平均梯度会相等
def te_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return torch.abs(te(A,A)-te(A,F))

# 与 MEFB 统一
def te_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    w0 = w1 = 1 # MEFB里边没有除 2
    q=1.85;     # Cvejic's constant
    return w0 * te(A, F, q, normalize=False) + w1 * te(B, F, q, normalize=False)

###########################################################################################

import matplotlib.pyplot as plt

def plot_scores(mi, nmi, te_values, te_labels):
    # Normalize scores
    def normalize_scores(scores):
        min_score = min(scores)
        max_score = max(scores)
        return [(score - min_score) / (max_score - min_score) for score in scores]

    mi = normalize_scores(mi)
    nmi = normalize_scores(nmi)
    te_values = {label: normalize_scores(te_values[label]) for label in te_labels}

    # combined_scores = list(zip(mi, nmi, te_values.items()))
    # combined_scores.sort(key=lambda x: x[0])
    # mi, nmi, te_values = zip(*combined_scores)

    # Plotting
    plt.plot(mi, label='MI')
    plt.plot(nmi, label='NMI')
    for label, te in te_values.items():
        plt.plot(te, label=f'TE a={label}')
    plt.legend()
    plt.show()

def demo():
    # Example usage
    mi = [1.2629, 1.0765, 0.7445, 1.3802, 1.1020, 0.9410, 2.8204, 1.7022, 2.4377, 1.2624, 1.2100, 1.0020, 1.2700]
    nmi = [0.2474, 0.2222, 0.1509, 0.2684, 0.2281, 0.1830, 0.5412, 0.3292, 0.4941, 0.2484, 0.2381, 0.2026, 0.2517]
    te_values = {
        '1.90': [251.9935, 277.1870, 32.7949, 33.3386, 316.3900, 87.2242, 35.7358, 68.2493, 355.0536, 240.9850, 514.4585, 43.5443, 33.7628],
        '1.85': [158.5506, 201.2620, 26.7743, 25.0559, 217.3696, 65.5279, 29.1573, 51.5961, 232.5418, 163.8177, 331.7446, 35.7764, 26.8387],
        '1.84': [144.8935, 188.9742, 25.7544, 23.7508, 201.9328, 61.9618, 28.0376, 48.8600, 214.4324, 151.9938, 304.3018, 34.4360, 25.6821],
        '1.83': [132.5332, 177.4989, 24.7868, 22.5402, 187.6841, 58.6145, 26.9743, 46.2922, 197.9685,141.1337,279.2694,33.1577,24.5904],
        '1.82': [121.3401, 166.7799, 23.8682, 21.4163, 174.5275, 55.4718, 25.9639, 43.8817, 182.9856,131.1532,256.4287,31.9383,23.5594],
        '1.81': [111.1976, 156.7649, 22.9956, 20.3718, 162.3751, 52.5203, 25.0036, 41.6181, 169.3370,121.9758,235.5817,30.7746,22.5853],
        '1.80': [102.0016, 147.4052, 22.1660, 19.4001, 151.1462, 49.7475, 24.0902, 39.4918, 156.8911,113.5319,216.5483,29.6636,21.6644],
        '1.75': [ 67.2589, 108.9667, 18.5785, 15.4420, 106.4727, 38.1841, 20.1387, 30.6267, 108.9946,80.2930,143.3684,24.8074,17.7486],
        '1.70': [ 45.5543, 81.3518,  15.7421, 12.5984,  76.0518, 29.6441, 17.0181, 24.0793,  77.8820, 57.9829, 96.4812, 20.9138, 14.7455],
        '1.65': [ 31.7428, 61.3731,  13.4675, 10.4995,  55.1232, 23.2843, 14.5223, 19.1987,  57.1430,42.7616,66.1477,17.7646,12.4106],
        '1.60': [ 22.7732, 46.8099,  11.6195,  8.9081,  40.5635, 18.5063, 12.5005, 15.5235,  42.9612,32.1961,46.3057,15.1958,10.5692],
        '1.50': [ 12.7609, 28.1757,   8.8364,  6.6808,  22.9916, 12.1065,  9.4621, 10.5683,  25.8800, 19.3492, 24.3284, 11.3324, 7.8998],
        '1.40': [  7.8802, 17.7502,   6.8699,  5.1986,  13.7777,  8.2689,  7.3042,  7.5474,  16.6915,12.4306,14.0915, 8.6378, 6.0887],
        '1.30': [  5.1681, 11.6450,   5.4099,  4.1162,   8.5532,  5.8383,  5.6640,  5.5817,  11.2728, 8.3497, 8.9177, 6.6932, 4.7743],
        '1.20': [  3.3225, 7.8030,    4.2336,  3.2230,   5.1401,  4.1503,  4.2540,  4.1625,   7.6882,5.5851,5.9679,5.2167,3.7164],
        '1.10': [  1.2662, 4.8169,    3.0044,  2.2047,   1.7485,  2.6042,  2.4985,  2.7935,   4.5143, 2.8554, 3.7264, 3.9056, 2.5717],
        '1.01': [-19.5154, -9.6679, -5.6413, -6.2604, -28.6412, -7.5607, -14.4913, -6.6281, -16.2042, -19.7543, -8.5319, -1.8701, -6.2008],
        '1.00': [-2.2440e+04,-1.4585e+04,-8.9908e+03,-8.9472e+03,-3.2598e+04,-1.0566e+04,-1.8206e+04,-9.8143e+03,-2.1567e+04,-2.4062e+04,-1.2568e+04,-5.6210e+03,-9.2046e+03]
        # Add more TE values as needed
    }
    te_labels = ['1.90', '1.85', '1.84', '1.83', '1.82', '1.81', '1.80', '1.75', '1.70',\
        '1.65', '1.60', '1.50', '1.40', '1.30', '1.20', '1.10', '1.01', '1.00']  # Example TE labels

    plot_scores(mi, nmi, te_values, te_labels)

def main():
    from torchvision import transforms
    from torchvision.transforms.functional import to_tensor
    from PIL import Image

    torch.manual_seed(42)

    transform = transforms.Compose([transforms.ToTensor()])

    vis = to_tensor(Image.open('../imgs/TNO/vis/9.bmp')).unsqueeze(0)
    ir = to_tensor(Image.open('../imgs/TNO/ir/9.bmp')).unsqueeze(0)
    fused = to_tensor(Image.open('../imgs/TNO/fuse/U2Fusion/9.bmp')).unsqueeze(0)

    # print(f'TE(ir,fused):{te(ir,fused)}')
    # print(f'TE(vis,fused):{te(vis,fused)}') # 73.67920684814453 正确
    # # print(f'TE(vis,fused):{te(vis,fused,normalize=True)}') # 48536.9453125错了
    # print(f'TE(fused,fused):{te(fused,fused)}')
    # print(f'TE(ir,ir):{te(ir,ir)}')
    # print(f'TE(vis,vis):{te(vis,vis)}')
    # print(f'TE_metric(ir,vis,fused):{te_metric(ir,vis,fused)}')

    demo()

if __name__ == '__main__':
    main()
