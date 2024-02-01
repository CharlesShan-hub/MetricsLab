import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import binom,norm,gaussian_kde,entropy
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import kornia

from sklearn.metrics.cluster import mutual_info_score as mi_sklearn

__all__ = [
    'entropy',       # 信息熵
    'kl_divergence', # kl散度
    'mi',            # 自己实现的mi
    'mi_sklearn'     # 不可微分mi，sklearn官方实现
]

def entropy(p):
    """
    计算给定概率分布的熵
    """
    entropy = 0
    for prob in p:
        if prob > 0:
            entropy += prob * np.log2(prob)
    return abs(entropy)

def kl_divergence(p, q):
    """
    计算两个离散概率分布之间的KL散度
    :param p: 第一个概率分布
    :param q: 第二个概率分布
    :return: KL散度值
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # 避免除以零的情况，将概率分布中的零值替换为一个小正数
    p[p == 0] = np.finfo(float).eps
    q[q == 0] = np.finfo(float).eps

    # 计算KL散度
    kl_div = np.sum(p * np.log(p / q))

    return kl_div


def mi(x, y, log=False):
    """
    根据两个序列计算互信息

    参数：
    x: 第一个序列
    y: 第二个序列

    返回值：
    互信息的值

    互信息的计算公式：
    I(X;Y) = H(X) + H(Y) - H(X,Y)

    其中，
    H(X) = - sum p(x) log_2 p(x)
    H(Y) = - sum p(y) log_2 p(y)
    H(X, Y) = - sum p(x, y) log_2 p(x, y)

    p(x), p(y) 为边缘概率分布，p(x, y) 为联合概率分布。
    """
    # 将序列转换为 NumPy 数组
    if log:
        print(f"Input x = {x}")
        print(f"Input x = {y}")
    x = np.array(x)
    x = x / x.sum()
    y = np.array(y)
    y = y / y.sum()
    if log:
        print(f"Px = {x}")
        print(f"Py = {y}")

    # 确保序列长度相同
    assert len(x) == len(y), "两个序列的长度不相同"

    # 计算联合概率分布
    Pxy = np.zeros((len(x), len(x)))
    for (i,x_i) in enumerate(x):
        for (j,y_j) in enumerate(y):
            Pxy[i,j] = x_i * y_j
    if log:
        print(f"Pxy = {Pxy}")

    hist, x_edges, y_edges = np.histogram2d(x, y, bins=(50, 50), range=[[0, 1], [0, 1]])


    # 计算边缘概率分布
    P_x = np.sum(Pxy, axis=1)
    P_y = np.sum(Pxy, axis=0)

    # 计算互信息
    mutual_information_value = 0
    for i in range(Pxy.shape[0]):
        for j in range(Pxy.shape[1]):
            if Pxy[i, j] > 0:
                mutual_information_value += Pxy[i, j] * np.log2(Pxy[i, j] / (P_x[i] * P_y[j]))

    return mutual_information_value

def mi(x, y, log=False):
    """
    根据两个序列计算互信息

    参数：
    x: 第一个序列
    y: 第二个序列

    返回值：
    互信息的值

    互信息的计算公式：
    I(X;Y) = H(X) + H(Y) - H(X,Y)

    其中，
    H(X) = - sum p(x) log_2 p(x)
    H(Y) = - sum p(y) log_2 p(y)
    H(X, Y) = - sum p(x, y) log_2 p(x, y)

    p(x), p(y) 为边缘概率分布，p(x, y) 为联合概率分布。
    """
    # 确保序列长度相同
    assert len(x) == len(y), "两个序列的长度不相同"

    # 计算联合概率分布
    p_joint, p_x1, p_x2 = np.histogram2d(x, y, bins=30, density=True)

    # 计算边缘概率分布
    #p_x1 = np.sum(p_joint, axis=1)
    #p_x2 = np.sum(p_joint, axis=0)

    # 计算边缘熵
    entropy_x1 = entropy(p_x1)
    entropy_x2 = entropy(p_x2)

    # 计算联合熵
    entropy_joint = entropy(p_joint.flatten())

    # 计算互信息
    mutual_info = entropy_x1 + entropy_x2 - entropy_joint

    return mutual_info



def mi_differentiable(image1, image2, bandwidth=0.25, eps=1e-10,normalize=False,show_pic=False):
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

    # 计算互信息
    mask = (hist > eps)
    en_xy = -torch.sum(hist[mask] * torch.log(hist[mask])) # VIFB里边用的 log，不是 log2
    mask = (marginal_x != 0)
    en_x = -torch.sum(marginal_x[mask] * torch.log(marginal_x[mask]))
    mask = (marginal_y != 0)
    en_y = -torch.sum(marginal_y[mask] * torch.log(marginal_y[mask]))

    if show_pic == True:
        hist_np, bin_edges_np = np.histogram(x1.numpy().flatten(), bins=256, range=[0, 256], density=True)
        plt.plot(bin_edges_np[:-1], hist_np, color='blue', label='Numpy Histogram')
        plt.plot(marginal_x.squeeze().detach().numpy(), color='orange', label='Kornia Histogram')
        plt.show()

    return en_x + en_y - en_xy


def mi_differentiable_loss(origin,predict):
    #return torch.exp(-mi_differentiable(x,y))
    #return torch.abs(1 - mi_differentiable(x,y) / mi_differentiable(x,x))
    return torch.abs(1 - mi_differentiable(origin,predict) / mi_differentiable(origin,origin))


def mi_metric(imgA,imgB,imgF):
    w0 = w1 = 1 # VIFB里边没有除 2
    return w0 * mi_differentiable(imgA,imgF) + w1 * mi_differentiable(imgB, imgF)


def demo_entropy():
    # 函数1
    def function1():
        # 生成 x 值
        x_values = np.linspace(0, 1, 100)

        # 计算信息熵
        entropies = [entropy([x, 1 - x]) for x in x_values]

        # 子图1
        plt.subplot(2, 2, 1)
        plt.plot(x_values, entropies)
        plt.xlabel('x')
        plt.ylabel('entropy')
        plt.title('Entropy of (x, 1-x)')

    # 函数2
    def function2():
        # 生成 x 和 y 的网格
        x_values = np.linspace(0, 1, 200)
        y_values = np.linspace(0, 1, 200)
        xx, yy = np.meshgrid(x_values, y_values)

        # 计算信息熵
        entropies = np.zeros_like(xx)
        for i in range(len(x_values)):
            for j in range(len(y_values)):
                entropies[i, j] = entropy([xx[i, j], yy[i, j]])

        # 子图2 - 绘制热图
        plt.subplot(2, 2, 2)

        # 设置透明度条件
        alpha_values = np.where(np.isclose(xx + yy, 1, atol=1e-2), 1, 0.2)

        plt.pcolormesh(xx, yy, entropies, cmap='viridis', alpha=alpha_values)

        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Entropy of (x, y)')

    # 函数3
    def function3():
        # 生成 x 和 y 的网格
        x_values = np.linspace(0, 1, 200)
        y_values = np.linspace(0, 1, 200)
        xx, yy = np.meshgrid(x_values, y_values)

        # 计算信息熵
        z_values = 1 - xx - yy
        entropies = np.zeros_like(z_values)
        alpha = np.zeros_like(z_values)
        for i in range(len(x_values)):
            for j in range(len(y_values)):
                # 如果 x + y 大于 1，透明度为 0，否则透明度为 1
                alpha[i,j] = 0 if xx[i, j] + yy[i, j] > 1 else 1
                entropies[i, j] = entropy([xx[i, j], yy[i, j], z_values[i, j]])

        # 子图3 - 绘制热图
        plt.subplot(2, 2, 3)

        plt.pcolormesh(xx, yy, entropies, cmap='viridis', alpha=alpha)

        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Entropy of (x, y, 1-x-y)')

    # 函数4
    def function4():
        ax = plt.subplot(2, 2, 4, projection='3d')
        ax.set_title("Entropy of (x, y, z)")
        # 生成 x, y 的坐标网格
        x_values = np.linspace(0, 1, 200)
        y_values = np.linspace(0, 1, 200)
        xx, yy = np.meshgrid(x_values, y_values)

        # 计算 z 坐标，满足 x + y + z = 1 且 z 大于零
        zz = np.maximum(0, 1 - xx - yy)

        # 过滤掉 z=0 的部分
        mask = zz > 0
        xx = np.where(mask, xx, np.nan)
        yy = np.where(mask, yy, np.nan)
        zz = np.where(mask, zz, np.nan)

        # 计算每个点的信息熵
        entropies = np.array([[entropy([x, y, z]) for x, y, z in zip(row_x, row_y, row_z)] for row_x, row_y, row_z in zip(xx, yy, zz)])

        # 绘制平面，使用信息熵来确定颜色
        surf = ax.plot_surface(xx, yy, zz, facecolors=plt.cm.viridis(entropies), rstride=5, cstride=5, alpha=0.8)
        surf.set_facecolor((0, 0, 0, 0))

        # 设置坐标轴标签
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    # 调用各个子图的函数绘制图形
    function1()
    function2()
    function3()
    function4()

    # 显示图形
    plt.show()


def demo_entropy2():
    def calculate_entropy_uniform(n):
        # 生成 n 个值为 1 的一维数组
        x = np.ones(n)

        # 归一化，使得和为 1
        x_normalized = x / np.sum(x)

        # 计算信息熵
        entropy_value = entropy(x_normalized)

        return entropy_value

    def calculate_entropy_single_one(n):
        # 生成 n 个值为 0 的一维数组
        x = np.zeros(n)

        # 将其中一个值设为 1
        x[0] = 1

        # 计算信息熵
        entropy_value = entropy(x)

        return entropy_value

    def calculate_entropy_mid(n):
        x0 = np.zeros(n)
        x0[0] = 1
        x1 = np.ones(n)
        x1 = x1 / x1.sum()
        x = (x0 + x1) / 2

        # 计算信息熵
        entropy_value = entropy(x)

        return entropy_value
    # 参数范围
    n_values = np.arange(1, 64*64)
    max_entropy = [calculate_entropy_uniform(n) for n in n_values]
    mix_entropy = [calculate_entropy_single_one(n) for n in n_values]
    mid_place_entropy = [calculate_entropy_mid(n) for n in n_values]
    per = [mid_value / max_value for (mid_value, max_value) in zip(mid_place_entropy, max_entropy)]

    # 绘制图表
    plt.figure(figsize=(5, 12))

    plt.subplot(2, 1, 1)
    plt.plot(n_values, max_entropy, label='Uniform Distribution')
    plt.plot(n_values, mix_entropy, label='Single 1, Rest 0s')
    plt.plot(n_values, mid_place_entropy, label='Mid of Center and Single 1')
    plt.xlabel('Parameter n')
    plt.ylabel('Entropy')
    plt.title('Entropy vs. n for Different Sequences')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(n_values, per)
    plt.xlabel('Parameter n')
    plt.ylabel('Entropy Ratio')
    plt.title(f'Entropy Ratio vs. n for 1/2 Distance')

    plt.tight_layout()
    plt.show()


def demo_kl_divergence():
    def uniform_distribution(n):
        s = np.ones(n)
        return s / np.sum(s)

    def binomial_distribution(n, p, size):
        # 生成二项分布
        binomial_dist = binom(n, p)
        # 得到权重数组 s2
        s2 = binomial_dist.pmf(np.arange(size))
        s2_normalized = s2 / np.sum(s2)
        return s2_normalized

    x = np.array([2, 3, 5, 14, 16, 15, 12, 8, 10, 8, 7])
    x = x / np.sum(x)

    # 使用 uniform 函数生成权重数组 s1
    s1 = uniform_distribution(len(x))

    # 使用 binomial 函数生成权重数组 s2
    s2 = binomial_distribution(n=10, p=0.5, size=len(x))

    # 归一化
    x_normalized = (x / np.sum(x)) * np.sum(s1)
    s1_normalized = s1 / np.sum(s1)

    # 创建左侧归一化后的柱状图
    plt.bar(np.arange(len(x)), x_normalized, color='blue', label='x', align='edge', width=0.4)

    # 创建右侧归一化后的柱状图（s1）
    plt.bar(np.arange(len(s1))+0.4, s1_normalized, color='orange', alpha=0.5, label=f'uniform:{kl_divergence(x,s1)}', align='edge', width=0.4)

    # 使用 s2 绘制归一化柱状图
    plt.bar(np.arange(len(s2))+0.8, s2, color='green', alpha=0.5, label=f'binomial:{kl_divergence(x,s2)}', align='edge', width=0.4)

    # 添加标签和标题
    plt.xlabel('Index')
    plt.ylabel('Normalized Value')
    plt.title('KL-Divergence Demo')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()


def demo_mi():
    def generate_and_plot_binomial_sequences(sequence_x1,sequence_x2):
        """
        生成并绘制两个二项分布示例序列的直方图和拟合曲线

        参数：
        n1: 第一个二项分布的试验次数
        p1: 第一个二项分布的成功概率
        n2: 第二个二项分布的试验次数
        p2: 第二个二项分布的成功概率
        size: 每个序列的长度
        """

        # 打印生成的序列
        print("Sequence X1:", sequence_x1)
        print("Sequence X2:", sequence_x2)

        # 计算直方图和拟合曲线的数据
        hist_data_x1, bins_x1 = np.histogram(sequence_x1, bins=np.arange(0, n1+2)-0.5, density=True)
        hist_data_x2, bins_x2 = np.histogram(sequence_x2, bins=np.arange(0, n2+2)-0.5, density=True)
        x1_range = np.linspace(bins_x1[0], bins_x1[-1], 100)
        x2_range = np.linspace(bins_x2[0], bins_x2[-1], 100)

        # 绘制直方图和拟合曲线
        plt.subplot(1,3,1)

        plt.bar(bins_x1[:-1], hist_data_x1, width=1, alpha=0.5, label=f'n={n1}, p={p1}', color='blue')
        plt.bar(bins_x2[:-1], hist_data_x2, width=1, alpha=0.5, label=f'n={n2}, p={p2}', color='orange')

        plt.plot(x1_range, norm.pdf(x1_range, np.mean(sequence_x1), np.std(sequence_x1)), color='blue', linestyle='dashed', label='Fit X1')
        plt.plot(x2_range, norm.pdf(x2_range, np.mean(sequence_x2), np.std(sequence_x2)), color='orange', linestyle='dashed', label='Fit X2')

        plt.xlabel('Values')
        plt.ylabel('Frequency / Probability Density')
        plt.legend()
        plt.title('Histogram and Fit of Binomial Distributions')

    def plot_joint_marginal_distributions(sequence_x1, sequence_x2):
        """
        根据给定的两个随机序列，绘制联合和边缘概率密度图

        参数：
        sequence_x1: 第一个随机序列
        sequence_x2: 第二个随机序列
        """
        # 计算联合概率密度和边缘概率密度
        H, xedges, yedges = np.histogram2d(sequence_x1, sequence_x2, bins=np.arange(0, n1+2)-0.5, density=True)
        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2

        # 计算边缘概率密度
        marginal_density_x1 = gaussian_kde(sequence_x1)(x_centers)
        marginal_density_x2 = gaussian_kde(sequence_x2)(y_centers)

        # 在同一图上绘制两个边缘概率密度
        #plt.subplot(2, 3, 5)
        #plt.hist(sequence_x1, bins=np.arange(0, n1+2)-0.5, density=True, color='blue', alpha=0.7)
        #plt.hist(sequence_x2, bins=np.arange(0, n2+2)-0.5, density=True, color='orange', alpha=0.7)
        #plt.plot(x_centers, marginal_density_x1, color='red', linestyle='dashed', label='KDE X1')
        #plt.plot(y_centers, marginal_density_x2, color='green', linestyle='dashed', label='KDE X2')
        #plt.xlabel('Values')
        #plt.ylabel('Density')
        #plt.legend()
        #plt.title('Marginal Probability Densities')

        # 绘制边缘概率密度图
        #plt.subplot(2, 3, 2)
        #plt.hist(sequence_x1, bins=np.arange(0, n1+2)-0.5, density=True, color='blue', alpha=0.7)
        #plt.plot(x_centers, marginal_density_x1, color='red', linestyle='dashed', label='KDE X1')
        #plt.xlabel('Sequence X1')
        #plt.ylabel('Density')
        #plt.legend()
        #plt.title('Marginal Probability Density X1')

        #plt.subplot(2, 3, 3)
        #plt.hist(sequence_x2, bins=np.arange(0, n2+2)-0.5, density=True, color='orange', alpha=0.7)
        #plt.plot(y_centers, marginal_density_x2, color='green', linestyle='dashed', label='KDE X2')
        #plt.xlabel('Density')
        #plt.ylabel('Sequence X2')
        #plt.legend()
        #plt.title('Marginal Probability Density X2')

        # 绘制联合概率密度图
        plt.subplot(1, 3, 2)
        plt.scatter(sequence_x1, sequence_x2, cmap='viridis', marker='.', alpha=0.3)
        plt.imshow(H, cmap='viridis', extent=[sequence_x1.min(), sequence_x1.max(), sequence_x2.min(), sequence_x2.max()])
        plt.colorbar()
        plt.xlabel('Sequence X1')
        plt.ylabel('Sequence X2')
        plt.title('Joint Probability Density')

        # 绘制MI
        plt.subplot(1,3,3)
        mi_value = mi(sequence_x1,sequence_x2)
        plt.imshow([[mi_value]], cmap='viridis', extent=[sequence_x1.min(), sequence_x1.max(), sequence_x2.min(), sequence_x2.max()])
        plt.colorbar(label='Mutual Information')
        plt.xticks([])
        plt.yticks([])
        plt.text(sequence_x1.mean(), sequence_x2.mean(), f'MI={mi_value:.2f}', color='white', ha='center', va='center', fontsize=12)
        plt.title('Mutual Information Heatmap')
        plt.show()

    n1 = 30; p1 = 0.7
    n2 = 45; p2 = 0.5
    size=5000

    # 生成两个示例序列
    sequence_x1 = np.random.binomial(n=n1, p=p1, size=size)
    sequence_x2 = np.random.binomial(n=n2, p=p2, size=size)

    # 第一个图，两个序列自己的直方图
    plt.figure(figsize=(10, 5))
    generate_and_plot_binomial_sequences(sequence_x1, sequence_x2)
    # 第二个图
    plot_joint_marginal_distributions(sequence_x1, sequence_x2)
    plt.show()


def demo_mi2():
    def binomial_entropy(n, p):
        # 计算二项分布的概率质量函数
        binomial_pmf = np.array([np.math.comb(n, k) * p**k * (1-p)**(n-k) for k in range(n+1)])

        # 计算熵
        entropy_value = entropy(binomial_pmf)

        return entropy_value

    def mutual_information_binomial(n, p):
        # 计算 B(n, p) 和 B(25, 0.75) 之间的互信息
        mi_value = binomial_entropy(n, p) + binomial_entropy(25, 0.75) - binomial_entropy(n+25, p*0.75)

        return mi_value
    # 参数范围
    n_values = np.arange(1, 100)
    p_values = np.arange(0.01, 0.99, 0.01)

    # 计算互信息矩阵
    mi_matrix = np.zeros((len(n_values), len(p_values)))

    for i, n in enumerate(n_values):
        for j, p in enumerate(p_values):
            mi_matrix[i, j] = mutual_information_binomial(n, p)

    # 绘制等高线
    levels = np.linspace(mi_matrix.min(), mi_matrix.max(), num=20)
    plt.contourf(p_values, n_values, mi_matrix, levels=levels, cmap='viridis')
    plt.colorbar(label='Mutual Information')

    # 标出特定点(n=25, p=0.75)
    plt.scatter(0.75, 25, color='red', marker='x', label='(25, 0.75)')
    plt.xlabel('Parameter p')
    plt.ylabel('Parameter n')
    plt.title('Mutual Information Contour for B(n, p) and B(25, 0.75)')
    plt.show()


def demo_mi3():
    # Generate integer values
    step = 32
    x_values = np.arange(0, 256, step)
    y_values = np.arange(0, 256, step)

    # Calculate mutual information values
    mutual_information1 = np.zeros((len(x_values), len(y_values)))
    mutual_information2 = np.zeros((len(x_values)**2, len(y_values)**2))
    mutual_information3 = np.zeros((len(x_values)**3, len(y_values)**3))
    mi_maxs = np.zeros((len(x_values)**2, len(y_values)**2))
    for i, x in enumerate(x_values):
        print(x)
        for j, y in enumerate(y_values):
            mutual_information1[i,j] = mi_sklearn([x],[y])
            for ii, xx in enumerate(x_values):
                for jj, yy in enumerate(y_values):
                    mutual_information2[i*len(x_values)+ii,j*len(y_values)+jj] = mi_sklearn([x,xx], [y,yy])
                    for iii, xxx in enumerate(x_values):
                        for jjj, yyy in enumerate(y_values):
                            mutual_information3[i*len(x_values)**2+ii*len(x_values)+iii,j*len(y_values)**2+jj*len(y_values)+jjj] = mi_sklearn([x,xx,xxx], [y,yy,yyy])


    # Plot mutual information heatmap
    plt.subplot(1,3,1)
    plt.imshow(mutual_information1, extent=(0, 255, 0, 255), origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='Mutual Information')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Mutual Information of [x] and [y]')

    # Plot mutual information heatmap
    plt.subplot(1,3,2)
    plt.imshow(mutual_information2, extent=(0, 255, 0, 255), origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='Mutual Information')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Mutual Information of [x,a] and [y,b]')

    # Plot mutual information heatmap
    plt.subplot(1,3,3)
    plt.imshow(mutual_information3, extent=(0, 255, 0, 255), origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='Mutual Information')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Mutual Information of [x,a,i] and [y,b,j]')
    plt.show()


def demo_mi4():
    # 生成坐标范围
    x_values = np.linspace(0.00, 0.99, 40)
    y_values = np.linspace(0.00, 0.99, 40)
    z_values = np.linspace(0.00, 0.99, 40)

    # epsilon 平滑值
    epsilon = 1e-8

    # 创建一个图形
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Mutual Information between X1 and X2 for Different Z')

    # 设置全局颜色映射范围
    vmin, vmax = -0.1, 2.5

    # 遍历每个 z 值
    for k, z in enumerate(z_values):
        # 初始化保存互信息的矩阵
        mi_matrix = np.zeros((len(x_values), len(y_values)))
        mi_max = np.zeros((len(x_values), len(y_values)))

        # 计算每个点对应的互信息
        for i, x in enumerate(x_values):
            for j, y in enumerate(y_values):
                # 检查坐标是否在 [0, 1] 范围内，否则跳过该点
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= z <= 1 and 0 <= y - z <= 1 and 0 <= x - z <= 1 and 0 <= 1 - y - x + z <= 1):
                    continue

                # 计算互信息，避免零概率导致的除零问题
                p_joint = np.array([[z, y - z], [x - z, 1 - y - x + z]])
                p_x1 = np.array([x, 1 - x])
                p_x2 = np.array([y, 1 - y])
                if p_joint[0,0]!= 0:
                    mi_matrix[i, j] += p_joint[0,0] * np.log(p_joint[0,0] / (p_x1[0]*p_x2[0]))
                if p_joint[0,1]!= 0:
                    mi_matrix[i, j] += p_joint[0,1] * np.log(p_joint[0,1] / (p_x1[0]*p_x2[1]))
                if p_joint[1,0]!= 0:
                    mi_matrix[i, j] += p_joint[1,0] * np.log(p_joint[1,0] / (p_x1[1]*p_x2[0]))
                if p_joint[1,1]!= 0:
                    mi_matrix[i, j] += p_joint[1,1] * np.log(p_joint[1,1] / (p_x1[1]*p_x2[1]))
                #mi_matrix[i, j] = np.sum(p_joint * np.log((p_joint + epsilon) / (np.outer(p_x1, p_x2) + epsilon)))
                mi_max = round(mi_matrix[i, j].max(),2)

        # 绘制二维图
        ax = fig.add_subplot(5, 8, k + 1)
        ax.set_title(f'Z = {z:.2f},Max={mi_max}')
        im = ax.imshow(mi_matrix, extent=(0, 1, 0, 1), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'Z = {z:.2f},Max={mi_max}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.colorbar(im, ax=ax, shrink=0.6)

    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 显示图形
    plt.show()


def demo_mi5():
    print("* 两随机图像的互信息：")
    for size in [64,128,256,512]:
        random_tensor1 = torch.randint(0, 256, size=(1, 1, size, size), dtype=torch.uint8)
        random_tensor2 = torch.randint(0, 256, size=(1, 1, size, size), dtype=torch.uint8)
        print(f" - {size} x {size} (self): ", mi_differentiable(random_tensor1,random_tensor2).item())
        print(f" - {size} x {size} (mklearn): ", mi_sklearn(random_tensor1.flatten().numpy(),random_tensor2.flatten().numpy()))
    print("* 随机图像与纯色图像的互信息：")
    for size in [64,128,256,512]:
        black_tensor = torch.ones(1, 1, size, size) * 0
        grey_tensor = torch.ones(1, 1, size, size) * 127
        white_tensor = torch.ones(1, 1, size, size) * 255
        random_tensor = torch.randint(0, 256, size=(1, 1, size, size), dtype=torch.uint8)
        print(f" - {size} x {size} (black,self): ", mi_differentiable(black_tensor,random_tensor).item())
        print(f" - {size} x {size} (black,mklearn): ", mi_sklearn(black_tensor.flatten().numpy(),random_tensor.flatten().numpy()))
        print(f" - {size} x {size} (grey,self): ", mi_differentiable(grey_tensor,random_tensor).item())
        print(f" - {size} x {size} (grey,mklearn): ", mi_sklearn(grey_tensor.flatten().numpy(),random_tensor.flatten().numpy()))
        print(f" - {size} x {size} (white,self): ", mi_differentiable(white_tensor,random_tensor).item())
        print(f" - {size} x {size} (white,mklearn): ", mi_sklearn(white_tensor.flatten().numpy(),random_tensor.flatten().numpy()))
    print("* (随机，随机)、(随机，可见光)互信息比较：")
    vis_tensor = TF.to_tensor(Image.open('../resources/imgs/vis/1.jpg')).unsqueeze(0)
    vis_tensor = torch.clamp(torch.mul(vis_tensor, 255), 0, 255).to(torch.uint8)
    random_tensor1 = torch.randint(0, 256, size=vis_tensor.shape, dtype=torch.uint8)
    random_tensor2 = torch.randint(0, 256, size=vis_tensor.shape, dtype=torch.uint8)
    total_elements = torch.prod(torch.tensor(vis_tensor.shape))# 计算张量的元素总数
    integer_tensor = torch.arange(256)# 使用torch.arange生成0到255的整数
    filled_tensor = integer_tensor.repeat(total_elements // 256 + 1)[:total_elements].view(vis_tensor.shape)
    shuffled_tensor = filled_tensor.flatten().gather(0, torch.randperm(total_elements)).view(vis_tensor.shape)
    print(f" - {vis_tensor.shape} (R,R,self): ", mi_differentiable(random_tensor1,random_tensor2).item())
    print(f" - {vis_tensor.shape} (R,Vis,self): ", mi_differentiable(random_tensor1,vis_tensor).item())
    print(f" - {vis_tensor.shape} (Uniform,Vis,self): ", mi_differentiable(filled_tensor,vis_tensor).item())
    print(f" - {vis_tensor.shape} (Uniform,Uniform,self): ", mi_differentiable(filled_tensor,filled_tensor).item())
    print(f" - {vis_tensor.shape} (Uniform,Uniform',self): ", mi_differentiable(filled_tensor,shuffled_tensor).item())
    print(f" - {vis_tensor.shape} (Uniform',Uniform',self): ", mi_differentiable(shuffled_tensor,shuffled_tensor).item())
    print(f" - {vis_tensor.shape} (Vis,Vis,self): ", mi_differentiable(vis_tensor,vis_tensor).item())
    print(f" - {vis_tensor.shape} (R,R,mklearn): ", mi_sklearn(random_tensor1.flatten().numpy(),random_tensor2.flatten().numpy()))
    print(f" - {vis_tensor.shape} (R,Vis,mklearn): ", mi_sklearn(random_tensor1.flatten().numpy(),vis_tensor.flatten().numpy()))
    print(f" - {vis_tensor.shape} (Uniform,Vis,mklearn): ", mi_sklearn(filled_tensor.flatten().numpy(),vis_tensor.flatten().numpy()))
    print(f" - {vis_tensor.shape} (Uniform,Uniform,mklearn): ", mi_sklearn(filled_tensor.flatten().numpy(),filled_tensor.flatten().numpy()))
    print(f" - {vis_tensor.shape} (Uniform,Uniform',mklearn): ", mi_sklearn(filled_tensor.flatten().numpy(),shuffled_tensor.flatten().numpy()))
    print(f" - {vis_tensor.shape} (Uniform',Uniform',mklearn): ", mi_sklearn(shuffled_tensor.flatten().numpy(),shuffled_tensor.flatten().numpy()))
    print(f" - {vis_tensor.shape} (Vis,Vis,mklearn): ", mi_sklearn(vis_tensor.flatten().numpy(),vis_tensor.flatten().numpy()))

def main():

  torch.manual_seed(42)  # 设置随机种子

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  img1_path = '../imgs/RoadScene/vis/1.jpg'
  img2_path = '../imgs/RoadScene/ir/1.jpg'
  fused_path = '../imgs/RoadScene/fuse/U2Fusion/1.jpg'

  transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
  )

  img1 = TF.to_tensor(Image.open(img1_path)).unsqueeze(0).to(device)
  img2 = TF.to_tensor(Image.open(img2_path)).unsqueeze(0).to(device)
  fused = TF.to_tensor(Image.open(fused_path)).unsqueeze(0).to(device)

  fused = fused.view(-1).cpu().numpy()
  img1 = img1.view(-1).cpu().numpy()
  img2 = img2.view(-1).cpu().numpy()

  # 测试 信息熵
  # 示例：假设有一个离散随机变量X，给定它的概率分布p(X)
  #cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  p_x = np.array([0.5, 0.5])
  p_y = np.array([0.3, 0.7])
  p_z = np.array([0.0, 1.0])
  print("* 信息熵")
  print(f" - E({p_x}) = {entropy(p_x)}")
  print(f" - E({p_y}) = {entropy(p_y)}")
  print(f" - E({p_z}) = {entropy(p_z)}")

  # 信息熵绘图案例
  #demo_entropy()
  #demo_entropy2()

  # 测试 kl_divergence
  p = [0.4, 0.3, 0.2, 0.1]
  q = [0.1, 0.2, 0.3, 0.4]
  print("* KL散度值")
  print(f" - p={p}, q={q}, KL散度值为: {kl_divergence(p, q)}")

  # kl_divergence案例
  #demo_kl_divergence()

  # 测试 自己实现的 MI 指标
  print("* MI")
  x = [0, 1, 0, 1, 0, 1]; y = [0, 1, 1, 0, 1, 0]
  print(f" - MI between {x} and {y}: {mi(x,y,log=True)}")
  x = [0, 1]; y = [1, 0]
  print(f" - MI between {x} and {y}: {mi(x,y,log=True)}")

  # 互信息案例
  #demo_mi()
  #demo_mi2()
  #demo_mi3()
  #demo_mi4()
  demo_mi5()

  # 测试 sklearn实现的 MI 指标
  result1 = mi_sklearn(fused, img1)
  result2 = mi_sklearn(fused, img2)
  print("* MI")
  print(f" - MI between Fused and Img1: {result1}")
  print(f" - MI between Fused and Img2: {result2}")

if __name__ == '__main__':

  main()
