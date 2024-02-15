from operator import mul
import torch
from torch._dynamo.compiled_autograd import maybe_clone
import kornia
import scipy
import matplotlib.pyplot as plt

###########################################################################################

__all__ = [
    'fmi',
    'fmi_approach_loss',
    'fmi_metric','fmi_p_metric','fmi_w_metric','fmi_e_metric','fmi_d_metric','fmi_g_metric',
]

def _wavelet(I, dwtEXTM='sym'):
    # Matlab:
    # [cA,cH,cV,cD] = dwt2(I,'dmey');
    # aFeature = rerange([cA,cH;cV,cD]);
    # M. Misiti, Y. Misiti, G. Oppenheim, J.M. Poggi 12-Mar-96.
    #
    # dwtEXTM: 'sym', 'per'(per是周期延拓的意思) 目前只写了sym方式
    try:
        W = torch.tensor(scipy.io.loadmat('../utils/dmey.mat')['dmey'])
    except:
        W = torch.tensor(scipy.io.loadmat('./utils/dmey.mat')['dmey'])
    Lo_R = W * torch.sqrt(torch.tensor(2))
    Hi_D = Lo_R.clone()
    Hi_D[:,0::2] *= -1
    Hi_R = torch.flip(Hi_D, [-2])
    Lo_D = torch.flip(Lo_R, [-2])

    L = Lo_R.shape[1]
    _ ,_, M, N = I.shape
    first = [2,2]  # 默认偏移量为 [0 0]，所以 first = [2 2]
    if dwtEXTM == 'sym':
        sizeEXT = L - 1
        last = [sizeEXT + M,sizeEXT + N]
    elif dwtEXTM == 'per': # 周期延拓模式
        sizeEXT = L // 2  #扩展大小为滤波器长度的一半
        last = [2 * ((M + 1) // 2), 2 * ((N + 1) // 2)]  # 周期延拓模式，最终大小为 2x2 的整数倍
    else:
        raise ValueError("dwtEXTM should be 'sym' or 'per'")

    def wextend_addcol(tensor, n):
        left_cols = torch.flip(tensor[:,:,:,:n], dims=[-1])
        right_cols = torch.flip(tensor[:,:,:,-n:], dims=[-1])
        return torch.cat([left_cols, tensor, right_cols], dim=-1)

    def wextend_addrow(tensor, n):
        up_raws = torch.flip(tensor[:,:,:n,:], dims=[-2])
        down_raws = torch.flip(tensor[:,:,-n:,:], dims=[-2])
        return torch.cat([up_raws, tensor, down_raws], dim=-2)

    def convdown(tensor, kernel, lenEXT):
        y = tensor[..., first[1]:last[1]:2]             # 提取需要进行卷积的子集
        y = wextend_addrow(y, lenEXT)   # 使用 'addrow' 模式对 y 进行扩展
        y = kornia.filters.filter2d(y.permute(0, 1, 3, 2),kernel.unsqueeze(0),padding="valid")
        y = y.permute(0, 1, 3, 2)[:,:,first[0]:last[0]:2,:]   # 提取结果的子集
        return y

    Y = wextend_addcol(I,sizeEXT)
    Z = kornia.filters.filter2d(Y,Lo_D.unsqueeze(0),padding="valid")
    A = convdown(Z,Lo_D,sizeEXT)
    H = convdown(Z,Hi_D,sizeEXT)
    Z = kornia.filters.filter2d(Y,Hi_D.unsqueeze(0),padding="valid")
    V = convdown(Z,Lo_D,sizeEXT)
    D = convdown(Z,Hi_D,sizeEXT)

    R = torch.cat([torch.cat([A, H], dim=-1),torch.cat([V, D], dim=-1)], dim=-2)
    return (R-torch.min(R)) / (torch.max(R)-torch.min(R))

def _gradient(I,eps=1e-10):
    # 使用Sobel算子计算水平和垂直梯度
    _grad_x = kornia.filters.filter2d(I,torch.tensor([[-1,  1]], dtype=torch.float64).unsqueeze(0))
    _grad_y = kornia.filters.filter2d(I,torch.tensor([[-1],[1]], dtype=torch.float64).unsqueeze(0))

    # 对梯度进行平均以避免过度敏感性(与 Matlab 统一)
    grad_x = (torch.cat((_grad_x[:,:,:,0:1],_grad_x[:,:,:,:-1]),dim=-1)+torch.cat((_grad_x[:,:,:,:-1],_grad_x[:,:,:,-2:-1]),dim=-1))/2
    grad_y = (torch.cat((_grad_y[:,:,0:1,:],_grad_y[:,:,:-1,:]),dim=-2)+torch.cat((_grad_y[:,:,:-1,:],_grad_y[:,:,-2:-1,:]),dim=-2))/2

    # 计算梯度的平均幅度
    s = torch.sqrt((grad_x ** 2 + grad_y ** 2 + eps)/2)

    #return grad_x # 在 matlab 代码中，没有求 y 方向的梯度，我进行了改进
    return s

def _edge(I, method='sobel',border_type='replicate', eps=1e-10): # matlab 版本默认是 sobel
    # 与 Matlab 的 [bx, by, b] = images.internal.builtins.edgesobelprewitt(a, isSobel, kx, ky); 完全一致
    grad_x = 1/8 * kornia.filters.filter2d(I,torch.tensor([[ 1,  0, -1],[ 2,  0, -2],[ 1,  0, -1]], dtype=torch.float64).unsqueeze(0),border_type=border_type)
    grad_y = 1/8 * kornia.filters.filter2d(I,torch.tensor([[ 1,  2,  1],[ 0,  0,  0],[-1, -2, -1]], dtype=torch.float64).unsqueeze(0),border_type=border_type)
    grad = (grad_x ** 2 + grad_y ** 2)
    # cutoff = scale * sum(b(:),'double') / numel(b); matlab中 scale 是 4
    # thresh = sqrt(cutoff);
    cutoff = 4 * torch.sum(grad) / (I.shape[-1] * I.shape[-2])
    thresh = torch.sqrt(cutoff)
    # 无法模拟
    # e = images.internal.builtins.computeEdges(b,bx,by,kx,ky,int8(offset),100*eps,cutoff);
    # 退而求其次: e = b > cutoff;
    e = (grad > cutoff).float()
    return e

def _dct(I):
    def dct(a):
        _, _, m, n = a.shape
        if n==1 or m==1:
            if n>1:
                do_trans = True
            else:
                do_trans = False
            a = torch.reshape(a,(1,1,m+n-1,1))
        else:
            do_trans = False
        _, _, n, m = a.shape

        aa = a

        if n % 2 == 1:
            y = torch.cat((aa, torch.flip(aa,[-2])),dim=-2)
            yy = torch.fft.fft(y, dim=-2)
            ww = (torch.exp(-1j * torch.arange(n) * torch.tensor(float(torch.pi) / (2 * n))) / torch.sqrt(torch.tensor(2 * n, dtype=torch.double))).unsqueeze(1)
            ww[0] = ww[0] / torch.sqrt(torch.tensor(2, dtype=torch.double))
            b = ww.expand(n, m)*(yy[:,:,:n,:])
        else:
            y = torch.cat((aa[:,:,::2, :], torch.flip(aa[:,:,1::2, :],[-2])), dim=-2)
            yy = torch.fft.fft(y, dim=-2)
            ww = 2 * torch.exp(-1j * torch.arange(n).unsqueeze(1) * torch.tensor(float(torch.pi) / (2 * n), dtype=torch.double)) / torch.sqrt(torch.tensor(2 * n, dtype=torch.double))
            ww[0] = ww[0] / torch.sqrt(torch.tensor(2, dtype=torch.double))
            b = ww.expand(n, m) * yy

        return b.real

    return torch.transpose(dct(torch.transpose(dct(I), dim0=-1, dim1=-2)), dim0=-1, dim1=-2)

class MIConv2d(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=0, kernel=None):
        super(MIConv2d, self).__init__()

    def forward(self, a_sub, f_sub):
        if torch.equal(a_sub, f_sub):
            return torch.tensor(1.0)
        else:
            return torch.tensor(0.0)
        '''
        def part_pdf(a_sub):
            a_max = torch.max(a_sub)
            a_min = torch.min(a_sub)
            if a_max == a_min:
                a_sub = torch.ones((2 * w + 1, 2 * w + 1))
            else:
                a_sub = (a_sub - a_min) / (a_max - a_min)
                torch.transpose(a_sub, dim0=-1, dim1=-2)
            a_pdf = torch.transpose(a_sub, dim0=-1, dim1=-2).flatten() / torch.sum(torch.sum(a_sub))
            return a_pdf
        if torch.equal(a_sub, f_sub):
            return torch.tensor(1)

        a_pdf = part_pdf(a_sub)
        f_pdf = part_pdf(f_sub)

        a_cdf = torch.zeros_like(a_pdf)
        f_cdf = torch.zeros_like(f_pdf)

        a_cdf[0] = a_pdf[0]
        f_cdf[0] = f_pdf[0]
        for i in range(1, l):
            a_cdf[i] = a_pdf[i] + a_cdf[i - 1]
            f_cdf[i] = f_pdf[i] + f_cdf[i - 1]

        a_temp = a_pdf - torch.mean(a_pdf)
        f_temp = f_pdf - torch.mean(f_pdf)
        if torch.sum(a_temp * f_temp) == 0:
            c = torch.tensor(0)
        else:
            c = torch.sum(a_temp * f_temp) / torch.sqrt(torch.sum(a_temp * a_temp) * torch.sum(f_temp * f_temp))

        e_a_pdf = torch.tensor(0.0)
        e2_a_pdf = torch.tensor(0.0)
        e_f_pdf = torch.tensor(0.0)
        e2_f_pdf = torch.tensor(0.0)
        for i in range(1, l + 1):
            e_a_pdf += i * a_pdf[i - 1]
            e2_a_pdf += (i ** 2) * a_pdf[i - 1]
            e_f_pdf += i * f_pdf[i - 1]
            e2_f_pdf += (i ** 2) * f_pdf[i - 1]
        a_sd = torch.sqrt(e2_a_pdf - e_a_pdf ** 2)
        f_sd = torch.sqrt(e2_f_pdf - e_f_pdf ** 2)

        joint_entropy = torch.tensor(0.0)

        if c >= 0:
            if c == 0 or a_sd == 0 or f_sd == 0:
                phi = torch.tensor(0.0)
            else:
                cov_up = 0
                for i in range(1, l + 1):
                    for j in range(1, l + 1):
                        cov_up += 0.5 * (f_cdf[i - 1] + a_cdf[j - 1] - torch.abs(f_cdf[i - 1] - a_cdf[j - 1])) - f_cdf[i - 1] * a_cdf[j - 1]
                corr_up = cov_up / (f_sd * a_sd)
                phi = c / corr_up

            jpdf_up = 0.5 * (f_cdf[0] + a_cdf[0] - torch.abs(f_cdf[0] - a_cdf[0]))
            jpdf = phi * jpdf_up + (1 - phi) * f_pdf[0] * a_pdf[0]

            if jpdf > 0:
                joint_entropy = -jpdf * torch.log2(jpdf)

            # 1-D boundaries
            for i in range(1, l):
                jpdf_up = 0.5 * (f_cdf[i] + a_cdf[0] - torch.abs(f_cdf[i] - a_cdf[0])) - \
                          0.5 * (f_cdf[i - 1] + a_cdf[0] - torch.abs(f_cdf[i - 1] - a_cdf[0]))
                jpdf = phi * jpdf_up + (1 - phi) * f_pdf[i] * a_pdf[0]
                if jpdf > 0:
                    joint_entropy += -jpdf * torch.log2(jpdf)  # 联合熵

            for j in range(1, l):
                jpdf_up = 0.5 * (f_cdf[0] + a_cdf[j] - torch.abs(f_cdf[0] - a_cdf[j])) - \
                          0.5 * (f_cdf[0] + a_cdf[j - 1] - torch.abs(f_cdf[0] - a_cdf[j - 1]))
                jpdf = phi * jpdf_up + (1 - phi) * f_pdf[0] * a_pdf[j]
                if jpdf > 0:
                    joint_entropy += -jpdf * torch.log2(jpdf)  # 联合熵

            # 2-D walls
            for i in range(1, l):
                for j in range(1, l):
                    jpdf_up = 0.5 * (f_cdf[i] + a_cdf[j] - torch.abs(f_cdf[i] - a_cdf[j])) - \
                              0.5 * (f_cdf[i - 1] + a_cdf[j] - torch.abs(f_cdf[i - 1] - a_cdf[j])) - \
                              0.5 * (f_cdf[i] + a_cdf[j - 1] - torch.abs(f_cdf[i] - a_cdf[j - 1])) + \
                              0.5 * (f_cdf[i - 1] + a_cdf[j - 1] - torch.abs(f_cdf[i - 1] - a_cdf[j - 1]))
                    jpdf = phi * jpdf_up + (1 - phi) * f_pdf[i] * a_pdf[j]
                    if jpdf > 0:
                        joint_entropy += -jpdf * torch.log2(jpdf)

        if c < 0:
            if a_sd == 0 or f_sd == 0:
                theta = torch.tensor(0)
            else:
                cov_lo = 0
                for i in range(1, l + 1):
                    for j in range(1, l + 1):
                        cov_lo += 0.5 * (f_cdf[i - 1] + a_cdf[j - 1] - 1 + torch.abs(f_cdf[i - 1] + a_cdf[j - 1] - 1)) - f_cdf[i - 1] * a_cdf[j - 1]
                corr_lo = cov_lo / (f_sd * a_sd)
                theta = c / corr_lo

            jpdf_lo = 0.5 * (f_cdf[0] + a_cdf[0] - 1 + torch.abs(f_cdf[0] + a_cdf[0] - 1))
            jpdf = theta * jpdf_lo + (1 - theta) * f_pdf[0] * a_pdf[0]
            if jpdf > 0:
                joint_entropy = -jpdf * torch.log2(jpdf)

            for i in range(1, l):
                jpdf_lo = 0.5 * (f_cdf[i] + a_cdf[0] - 1 + torch.abs(f_cdf[i] + a_cdf[0] - 1)) - \
                            0.5 * (f_cdf[i - 1] + a_cdf[0] - 1 + torch.abs(f_cdf[i - 1] + a_cdf[0] - 1))
                jpdf = theta * jpdf_lo + (1 - theta) * f_pdf[i] * a_pdf[0]
                if jpdf > 0:
                    joint_entropy += -jpdf * torch.log2(jpdf)

            for j in range(1, l):
                jpdf_lo = 0.5 * (f_cdf[0] + a_cdf[j] - 1 + torch.abs(f_cdf[0] + a_cdf[j] - 1)) - \
                            0.5 * (f_cdf[0] + a_cdf[j - 1] - 1 + torch.abs(f_cdf[0] + a_cdf[j - 1] - 1))
                jpdf = theta * jpdf_lo + (1 - theta) * f_pdf[0] * a_pdf[j]
                if jpdf > 0:
                    joint_entropy += -jpdf * torch.log2(jpdf)

            for i in range(1, l):
                for j in range(1, l):
                    jpdf_lo = 0.5 * (f_cdf[i] + a_cdf[j] - 1 + torch.abs(f_cdf[i] + a_cdf[j] - 1)) - \
                                0.5 * (f_cdf[i - 1] + a_cdf[j] - 1 + torch.abs(f_cdf[i - 1] + a_cdf[j] - 1)) - \
                                0.5 * (f_cdf[i] + a_cdf[j - 1] - 1 + torch.abs(f_cdf[i] + a_cdf[j - 1] - 1)) + \
                                0.5 * (f_cdf[i - 1] + a_cdf[j - 1] - 1 + torch.abs(f_cdf[i - 1] + a_cdf[j - 1] - 1))
                    jpdf = theta * jpdf_lo + (1 - theta) * f_pdf[i] * a_pdf[j]
                    if jpdf > 0:
                        joint_entropy += -jpdf * torch.log2(jpdf)

        index = torch.nonzero(a_pdf > 0).squeeze()
        a_entropy = torch.sum(-a_pdf[index] * torch.log2(a_pdf[index]))
        index = torch.nonzero(f_pdf > 0).squeeze()
        f_entropy = torch.sum(-f_pdf[index] * torch.log2(f_pdf[index]))

        # Mutual information between a & f
        mi = a_entropy + f_entropy - joint_entropy

        # Overall normalized mutual information
        if mi == 0:
            return torch.tensor(0.0)
        else:
            return 2 * mi / (a_entropy + f_entropy)
    '''



def fmi(A, B, F, feature='pixel', window_size=3):
    # feature: 'gradient', 'edge', 'dct', 'wavelet', 'pixel'
    if feature == 'pixel':
        [A,B,F] = [A*255,B*255,F*255]
    elif feature == 'wavelet':
        [A,B,F] = [_wavelet(I*255) for I in [A, B, F]]
    elif feature == 'dct':
        [A,B,F] = [_dct(I*255) for I in [A, B, F]]
    elif feature == 'edge': # 未完全复现，没有按照 thin 的方案
        [A,B,F] = [_edge(I*255) for I in [A, B, F]]
    elif feature == 'gradient':
        [A,B,F] = [_gradient(I*255) for I in [A, B, F]]
    else:
        raise ValueError("feature should be: 'gradient', 'edge', 'dct', 'wavelet', 'pixel'")

    _, _, m, n = A.shape
    w = int((window_size+1)/2-1)
    fmi_map = torch.ones((m - 2 * w, n - 2 * w))

    # 展开
    def unfolded_image(I):
        unfolded = torch.nn.functional.unfold(I, (m-2*w,n-2*w), stride=1)
        return unfolded.view(unfolded.size(0), unfolded.size(1), -1).squeeze(0)  # 将展平的张量重新变形为窗口数据
    [uA, uB, uF] = [unfolded_image(I) for I in [A, B, F]]

    # 计算相等的区域
    def cal_same(uI, uF):
        return (~(uI == uF).all(dim=-1)).float().unsqueeze(-1)
    [sameAF, sameBF] = [cal_same(uI, uF) for uI in [uA, uB]]

    # 计算pdf
    def cal_pdf(uI):
        # 计算最大最小值
        (max_uI,_) = torch.max(uI,dim=1)
        (min_uI,_) = torch.min(uI,dim=1)
        max_uI = max_uI.unsqueeze(-1)
        min_uI = min_uI.unsqueeze(-1)
        # 调整分母
        denominator = max_uI - min_uI
        denominator[max_uI == min_uI] = 1
        # 调整分子
        classify = denominator.clone()
        classify[max_uI != min_uI] = 0
        classify = torch.matmul(classify,torch.ones(1,uI.shape[-1]))
        numerator = uI - min_uI
        numerator = (1-classify) * numerator + classify
        # 规范化
        uI = numerator / denominator
        # pdf
        return uI / torch.sum(uI,dim=1).unsqueeze(-1)
    [pdfA, pdfB, pdfF] = [cal_pdf(uI) for uI in [uA, uB, uF]]

    # 累积求和得到 CDF
    [cdfA, cdfB, cdfF] = [torch.cumsum(pdfI, dim=1) for pdfI in [pdfA, pdfB, pdfF]]
    # print(cdfA)

    # 计算判别常数 C(与 0 的大小比较)
    def cal_c(pdfI, pdfF):
        tempI = pdfI - torch.mean(pdfI,dim=1).unsqueeze(-1)
        tempF = pdfF - torch.mean(pdfF,dim=1).unsqueeze(-1)
        sumIF = torch.sum(tempI * tempF, dim=1)
        sumII = torch.sum(tempI * tempI, dim=1)
        sumFF = torch.sum(tempF * tempF, dim=1)
        c = (sumIF / torch.sqrt(sumII * sumFF)).unsqueeze(-1)
        return (c<0).float()
    [cAF, cBF] = [cal_c(pdfI,pdfF) for pdfI in [pdfA, pdfB]]

    # 标准差
    def cal_sd(pdfI):
        weight = torch.arange(1,window_size**2+1,1)
        pdfEI = torch.sum(weight * pdfI,dim=1).unsqueeze(-1)
        pdfE2I = torch.sum(weight**2 * pdfI,dim=1).unsqueeze(-1)
        return torch.sqrt(pdfE2I - pdfEI**2)
    [sdA, sdB, sdF] = [cal_sd(pdfI) for pdfI in [pdfA, pdfB, pdfF]]

    # phi
    def cal_phi(cdfI,cdfF,sdI,sdF):
        pass





    def part_pdf(a_sub):
        a_max = torch.max(a_sub)
        a_min = torch.min(a_sub)
        if a_max == a_min:
            a_sub = torch.ones((2 * w + 1, 2 * w + 1))
        else:
            a_sub = (a_sub - a_min) / (a_max - a_min)
            torch.transpose(a_sub, dim0=-1, dim1=-2)
        a_pdf = torch.transpose(a_sub, dim0=-1, dim1=-2).flatten() / torch.sum(torch.sum(a_sub))
        return a_pdf
        if torch.equal(a_sub, f_sub):
            return torch.tensor(1)

    def part(a_sub,f_sub):
        a_pdf = part_pdf(a_sub)
        f_pdf = part_pdf(f_sub)

        a_cdf = torch.zeros_like(a_pdf)
        f_cdf = torch.zeros_like(f_pdf)

        a_cdf[0] = a_pdf[0]
        f_cdf[0] = f_pdf[0]
        for i in range(1, l):
            a_cdf[i] = a_pdf[i] + a_cdf[i - 1]
            f_cdf[i] = f_pdf[i] + f_cdf[i - 1]

        print(a_cdf)
        a_temp = a_pdf - torch.mean(a_pdf)
        f_temp = f_pdf - torch.mean(f_pdf)
        if torch.sum(a_temp * f_temp) == 0:
            c = torch.tensor(0)
        else:
            c = torch.sum(a_temp * f_temp) / torch.sqrt(torch.sum(a_temp * a_temp) * torch.sum(f_temp * f_temp))

        e_a_pdf = torch.tensor(0.0)
        e2_a_pdf = torch.tensor(0.0)
        e_f_pdf = torch.tensor(0.0)
        e2_f_pdf = torch.tensor(0.0)
        for i in range(1, l + 1):
            e_a_pdf += i * a_pdf[i - 1]
            e2_a_pdf += (i ** 2) * a_pdf[i - 1]
            e_f_pdf += i * f_pdf[i - 1]
            e2_f_pdf += (i ** 2) * f_pdf[i - 1]
        a_sd = torch.sqrt(e2_a_pdf - e_a_pdf ** 2)
        f_sd = torch.sqrt(e2_f_pdf - e_f_pdf ** 2)

        joint_entropy = torch.tensor(0.0)

        if c >= 0:
            if c == 0 or a_sd == 0 or f_sd == 0:
                phi = torch.tensor(0.0)
            else:
                cov_up = 0
                count = 0
                for i in range(1, l + 1):
                    for j in range(1, l + 1):
                        cov_up += 0.5 * (f_cdf[i - 1] + a_cdf[j - 1] - torch.abs(f_cdf[i - 1] - a_cdf[j - 1])) - f_cdf[i - 1] * a_cdf[j - 1]
                        count += 1
                print(count)
                corr_up = cov_up / (f_sd * a_sd)
                phi = c / corr_up

            jpdf_up = 0.5 * (f_cdf[0] + a_cdf[0] - torch.abs(f_cdf[0] - a_cdf[0]))
            jpdf = phi * jpdf_up + (1 - phi) * f_pdf[0] * a_pdf[0]
            print(jpdf)
            if jpdf > 0:
                joint_entropy = -jpdf * torch.log2(jpdf)

            # 1-D boundaries
            for i in range(1, l):
                jpdf_up = 0.5 * (f_cdf[i] + a_cdf[0] - torch.abs(f_cdf[i] - a_cdf[0])) - \
                          0.5 * (f_cdf[i - 1] + a_cdf[0] - torch.abs(f_cdf[i - 1] - a_cdf[0]))
                jpdf = phi * jpdf_up + (1 - phi) * f_pdf[i] * a_pdf[0]
                if jpdf > 0:
                    joint_entropy += -jpdf * torch.log2(jpdf)  # 联合熵

            for j in range(1, l):
                jpdf_up = 0.5 * (f_cdf[0] + a_cdf[j] - torch.abs(f_cdf[0] - a_cdf[j])) - \
                          0.5 * (f_cdf[0] + a_cdf[j - 1] - torch.abs(f_cdf[0] - a_cdf[j - 1]))
                jpdf = phi * jpdf_up + (1 - phi) * f_pdf[0] * a_pdf[j]
                if jpdf > 0:
                    joint_entropy += -jpdf * torch.log2(jpdf)  # 联合熵

            # 2-D walls
            for i in range(1, l):
                for j in range(1, l):
                    jpdf_up = 0.5 * (f_cdf[i] + a_cdf[j] - torch.abs(f_cdf[i] - a_cdf[j])) - \
                              0.5 * (f_cdf[i - 1] + a_cdf[j] - torch.abs(f_cdf[i - 1] - a_cdf[j])) - \
                              0.5 * (f_cdf[i] + a_cdf[j - 1] - torch.abs(f_cdf[i] - a_cdf[j - 1])) + \
                              0.5 * (f_cdf[i - 1] + a_cdf[j - 1] - torch.abs(f_cdf[i - 1] - a_cdf[j - 1]))
                    jpdf = phi * jpdf_up + (1 - phi) * f_pdf[i] * a_pdf[j]
                    if jpdf > 0:
                        joint_entropy += -jpdf * torch.log2(jpdf)

        if c < 0:
            if a_sd == 0 or f_sd == 0:
                theta = torch.tensor(0)
            else:
                cov_lo = 0
                for i in range(1, l + 1):
                    for j in range(1, l + 1):
                        cov_lo += 0.5 * (f_cdf[i - 1] + a_cdf[j - 1] - 1 + torch.abs(f_cdf[i - 1] + a_cdf[j - 1] - 1)) - f_cdf[i - 1] * a_cdf[j - 1]
                corr_lo = cov_lo / (f_sd * a_sd)
                theta = c / corr_lo

            jpdf_lo = 0.5 * (f_cdf[0] + a_cdf[0] - 1 + torch.abs(f_cdf[0] + a_cdf[0] - 1))
            jpdf = theta * jpdf_lo + (1 - theta) * f_pdf[0] * a_pdf[0]
            print(jpdf)
            if jpdf > 0:
                joint_entropy = -jpdf * torch.log2(jpdf)

            for i in range(1, l):
                jpdf_lo = 0.5 * (f_cdf[i] + a_cdf[0] - 1 + torch.abs(f_cdf[i] + a_cdf[0] - 1)) - \
                            0.5 * (f_cdf[i - 1] + a_cdf[0] - 1 + torch.abs(f_cdf[i - 1] + a_cdf[0] - 1))
                jpdf = theta * jpdf_lo + (1 - theta) * f_pdf[i] * a_pdf[0]
                if jpdf > 0:
                    joint_entropy += -jpdf * torch.log2(jpdf)

            for j in range(1, l):
                jpdf_lo = 0.5 * (f_cdf[0] + a_cdf[j] - 1 + torch.abs(f_cdf[0] + a_cdf[j] - 1)) - \
                            0.5 * (f_cdf[0] + a_cdf[j - 1] - 1 + torch.abs(f_cdf[0] + a_cdf[j - 1] - 1))
                jpdf = theta * jpdf_lo + (1 - theta) * f_pdf[0] * a_pdf[j]
                if jpdf > 0:
                    joint_entropy += -jpdf * torch.log2(jpdf)

            for i in range(1, l):
                for j in range(1, l):
                    jpdf_lo = 0.5 * (f_cdf[i] + a_cdf[j] - 1 + torch.abs(f_cdf[i] + a_cdf[j] - 1)) - \
                                0.5 * (f_cdf[i - 1] + a_cdf[j] - 1 + torch.abs(f_cdf[i - 1] + a_cdf[j] - 1)) - \
                                0.5 * (f_cdf[i] + a_cdf[j - 1] - 1 + torch.abs(f_cdf[i] + a_cdf[j - 1] - 1)) + \
                                0.5 * (f_cdf[i - 1] + a_cdf[j - 1] - 1 + torch.abs(f_cdf[i - 1] + a_cdf[j - 1] - 1))
                    jpdf = theta * jpdf_lo + (1 - theta) * f_pdf[i] * a_pdf[j]
                    if jpdf > 0:
                        joint_entropy += -jpdf * torch.log2(jpdf)

        index = torch.nonzero(a_pdf > 0).squeeze()
        a_entropy = torch.sum(-a_pdf[index] * torch.log2(a_pdf[index]))
        index = torch.nonzero(f_pdf > 0).squeeze()
        f_entropy = torch.sum(-f_pdf[index] * torch.log2(f_pdf[index]))

        # Mutual information between a & f
        mi = a_entropy + f_entropy - joint_entropy

        # Overall normalized mutual information
        if mi == 0:
            return torch.tensor(0.0)
        else:
            return 2 * mi / (a_entropy + f_entropy)

    for p in range(w, m - w):
        for q in range(w, n - w):
            a_sub = A[:,:,p - w:p + w + 1, q - w:q + w + 1]
            b_sub = B[:,:,p - w:p + w + 1, q - w:q + w + 1]
            f_sub = F[:,:,p - w:p + w + 1, q - w:q + w + 1]
            l = (2 * w + 1) ** 2
            fmi_af = part(a_sub,f_sub)
            fmi_bf = part(b_sub,f_sub)
            fmi_map[p-w,q-w] = (fmi_af + fmi_bf)/2;
            # print(fmi_af,fmi_bf)
            if q==w+1:
                raise

    return torch.mean(fmi_map)


def fmi_approach_loss():
    pass

def fmi_metric(A, B, F, feature='pixel', window_size=3):
    return fmi(A, B, F, feature, window_size)

def fmi_p_metric(A, B, F):
    return fmi(A, B, F, feature='pixel', window_size=3)

def fmi_w_metric(A, B, F):
    return fmi(A, B, F, feature='wavelet', window_size=3)

def fmi_e_metric(A, B, F):
    return fmi(A, B, F, feature='edge', window_size=3)

def fmi_d_metric(A, B, F):
    return fmi(A, B, F, feature='dct', window_size=3)

def fmi_g_metric(A, B, F):
    return fmi(A, B, F, feature='gradient', window_size=3)


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
    toy1 = torch.tensor([[[[1,1,1,1,1,1],
                           [1,0,0,0,0,0],
                           [2,1,0,0,0,0],
                           [2,1,0,0,0,0],
                           [2,1,0,0,0,0]]]])*0.5
    toy2 = torch.tensor([[[[1,1,1,1,1,1],
                           [1,0,0,0,1,1],
                           [2,1,0,0,0,0],
                           [2,1,0,0,0,2],
                           [2,1,0,0,0,0]]]])*0.5

    # print(f'FMI(pixel):{fmi(vis,ir,fused,feature="pixel")}')
    # print(f'FMI(wavelet):{fmi(vis,ir,fused,feature="wavelet")}')
    # print(f'FMI(dct):{fmi(vis,ir,fused,feature="dct")}')
    # print(f'FMI(edge):{fmi(vis,ir,fused,feature="edge")}')
    # print(f'FMI(gradient):{fmi(vis,ir,fused,feature="gradient")}')
    #fmi(toy1,toy1,toy2,feature="pixel")
    # fmi(toy1,toy1,toy2)
    fmi(vis,ir,fused,feature="pixel")


if __name__ == '__main__':
    main()
