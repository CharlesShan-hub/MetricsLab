import torch
import torch.fft
import kornia

def qcb(im1, im2, fused):
    im1 = im1.double()
    im2 = im2.double()
    fused = fused.double()

    im1 = normalize1(im1)
    im2 = normalize1(im2)
    fused = normalize1(fused)

    f0 = 15.3870
    f1 = 1.3456
    a = 0.7622

    k = 1
    h = 1
    p = 3
    q = 2
    Z = 0.0001
    sigma = 2
    batch_size = 1

    _, _, hang, lie = im1.size()

    meshgrid = kornia.create_meshgrid(hang, lie, normalized_coordinates=False)
    u, v = meshgrid[0, :, :, 0], meshgrid[0, :, :, 1]
    r = torch.sqrt(u**2 + v**2)

    Sd = torch.exp(-(r / f0)**2) - a * torch.exp(-(r / f1)**2)

    fused1 = torch.fft.ifftn(torch.fft.fftn(im1) * Sd)
    fused2 = torch.fft.ifftn(torch.fft.fftn(im2) * Sd)
    ffused = torch.fft.ifftn(torch.fft.fftn(fused) * Sd)

    G1 = kornia.filters.gaussian((hang, lie), torch.tensor([[2]] * batch_size, dtype=torch.float32))
    G2 = kornia.filters.gaussian((hang, lie), torch.tensor([[4]] * batch_size, dtype=torch.float32))

    C1 = contrast(G1, G2, fused1)
    C1 = torch.abs(C1)
    C1P = (k * (C1**p)) / (h * (C1**q) + Z)

    C2 = contrast(G1, G2, fused2)
    C2 = torch.abs(C2)
    C2P = (k * (C2**p)) / (h * (C2**q) + Z)

    Cf = contrast(G1, G2, ffused)
    Cf = torch.abs(Cf)
    CfP = (k * (Cf**p)) / (h * (Cf**q) + Z)

    mask = (C1P < CfP).double()
    Q1F = (C1P / CfP) * mask + (CfP / C1P) * (1 - mask)

    mask = (C2P < CfP).double()
    Q2F = (C2P / CfP) * mask + (CfP / C2P) * (1 - mask)

    ramda1 = (C1P**2) / (C1P**2 + C2P**2)
    ramda2 = (C2P**2) / (C1P**2 + C2P**2)

    Q = ramda1 * Q1F + ramda2 * Q2F

    return Q.mean(dim=(1, 2)).item()

def gaussian2d(n1, n2, sigma):
    H = (n1 - 1) // 2
    L = (n2 - 1) // 2

    x, y = kornia.create_meshgrid(2 * 15 + 1, 2 * 15 + 1, normalized_coordinates=False)
    G = torch.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * sigma**2 * torch.pi)

    return G

def contrast(G1, G2, im):
    buff = F.conv2d(im.unsqueeze(0).unsqueeze(0), G1.unsqueeze(0).unsqueeze(0), padding=G1.size(0) // 2)
    buff1 = F.conv2d(im.unsqueeze(0).unsqueeze(0), G2.unsqueeze(0).unsqueeze(0), padding=G2.size(0) // 2)

    return buff.squeeze() / buff1.squeeze() - 1

def normalize1(data):
    data = data.double()
    da = torch.max(data)
    xiao = torch.min(data)
    if da == 0 and xiao == 0:
        return data
    else:
        newdata = (data - xiao) / (da - xiao)
        return torch.round(newdata * 255)

# 示例用法
img1 = torch.rand((1, 1, 256, 256))
img2 = torch.rand((1, 1, 256, 256))
fused = torch.rand((1, 1, 256, 256))
result = qcb(img1, img2, fused)
print(result)
