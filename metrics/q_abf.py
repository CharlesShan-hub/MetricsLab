import torch
import kornia
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

def has_nan_or_has_inf(tensor):
  # 检查张量中是否有 NaN
  has_nan = torch.isnan(tensor)
  print("Has NaN:", has_nan.any().item())

  # 检查张量中是否有无穷值
  has_inf = torch.isinf(tensor)
  print("Has Inf:", has_inf.any().item())

''' Q_ABF
  * border_type = 'constant', 为了与 VIFB 一致，但其实 kornia 默认的是 'reflection'
'''
def q_abf(imgA, imgB, imgF, border_type='constant', eps=1e-6):
    # 参数
    Tg, kg, Dg = 0.9994, -15, 0.5
    Ta, ka, Da = 0.9879, -22, 0.8
    #print("1. Image Mean")
    #print(torch.mean(imgA),torch.mean(imgB),torch.mean(imgF))

    # 边缘强度和方向
    def edge_strength_and_orientation(tensor):
        gx = kornia.filters.filter2d(tensor,torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float64).unsqueeze(0), border_type=border_type)
        gy = kornia.filters.filter2d(tensor,torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float64).unsqueeze(0), border_type=border_type)
        #magnitude = torch.sqrt(gx * gx + gy * gy) # Before
        magnitude = torch.sqrt(gx * gx + gy * gy + eps) # Improve - This Works!
        orientation = torch.atan(gy/(gx+eps))
        #orientation[gx == 0] = torch.pi / 2 # Before
        orientation[torch.abs(gx) < eps] = torch.pi / 2 # Improve
        return (magnitude,orientation)
        #return (magnitude,magnitude)

    (gA,aA) = edge_strength_and_orientation(imgA*255.0)
    (gB,aB) = edge_strength_and_orientation(imgB*255.0)
    (gF,aF) = edge_strength_and_orientation(imgF*255.0)
    #return torch.mean(gA)+torch.mean(gB)-2*torch.mean(gF) + torch.mean(aA)+torch.mean(aB)-2*torch.mean(aF)
    #has_nan_or_has_inf(gA)
    #has_nan_or_has_inf(aA)
    #has_nan_or_has_inf(gB)
    #has_nan_or_has_inf(aB)
    #has_nan_or_has_inf(gF)
    #has_nan_or_has_inf(aF)
    #print("2. Edge Strength and Orientation")
    #print(torch.mean(gA),torch.mean(aA),torch.mean(gB),torch.mean(aB),torch.mean(gF),torch.mean(aF))

    # 相对强度和方向值
    def relative_strength_and_orientation(gA,gB,aA,aB):
        #g = torch.where(gA != gB, torch.minimum(gA, gB) / torch.maximum(gA, gB), gA)
        g_denom = torch.where(gA != gB, torch.maximum(gA, gB), gA)         # Improve
        g_denom = torch.clamp(g_denom, min=eps)                            # Improve
        g = torch.where(g_denom != 0, torch.minimum(gA, gB) / g_denom, gA) # Improve
        a = 1 - torch.abs(aA - aB) / (torch.pi / 2)
        return (g,a)

    (GAF,AAF) = relative_strength_and_orientation(gA,gF,aA,aF)
    (GBF,ABF) = relative_strength_and_orientation(gB,gF,aB,aF)
    #has_nan_or_has_inf(GAF)
    #has_nan_or_has_inf(AAF)
    #has_nan_or_has_inf(GBF)
    #has_nan_or_has_inf(ABF)
    #print("3. Relative Strength and Orientation")
    #print(torch.mean(GAF),torch.mean(AAF),torch.mean(GBF),torch.mean(ABF))

    # 边缘强度和方向保留值
    def edge_strength_and_orientation_preservation_values(G_F,A_F):
        #Qg = Tg / (1 + torch.exp(kg * (G_F - Dg)))
        #Qa = Ta / (1 + torch.exp(ka * (A_F - Da)))
        Qg = Tg / (1 + torch.exp(torch.clamp(kg * (G_F - Dg), min=-20, max=20))) # Improve
        Qa = Ta / (1 + torch.exp(torch.clamp(ka * (A_F - Da), min=-20, max=20))) # Improve
        return Qg * Qa

    QAF = edge_strength_and_orientation_preservation_values(GAF,AAF)
    QBF = edge_strength_and_orientation_preservation_values(GBF,ABF)
    #has_nan_or_has_inf(QAF)
    #has_nan_or_has_inf(QBF)
    #print("4. Edge Strength and Orientation Preservation Values")
    #print(torch.mean(QAF),torch.mean(QBF))

    deno = torch.sum(gA + gB)
    nume = torch.sum(QAF * gA + QBF * gB)
    qabf_value = nume / deno
    #has_nan_or_has_inf(qabf_value)

    return qabf_value


def q_abf_loss(imgA, imgB, imgF):
    return -q_abf(imgA, imgB, imgF)


def q_abf_metric(imgA,imgB,imgF):
    return q_abf(imgA, imgB, imgF)


def main():
    torch.manual_seed(42)  # 设置随机种子

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img1_path = '../imgs/TNO/vis/9.bmp'
    img2_path = '../imgs/TNO/ir/9.bmp'
    fused_path = '../imgs/TNO/fuse/U2Fusion/9.bmp'

    transform = transforms.Compose(
      [
          transforms.ToTensor(),
      ]
    )

    img1 = TF.to_tensor(Image.open(img1_path)).unsqueeze(0).to(device)
    img2 = TF.to_tensor(Image.open(img2_path)).unsqueeze(0).to(device)
    fused = TF.to_tensor(Image.open(fused_path)).unsqueeze(0).to(device)

    print(q_abf(img1,img2,fused))
    print(q_abf(img1,img1,img1))

    # 逐个检查小函数的梯度
    #check_gradients(q_abf, (img1,img2,fused))


if __name__ == '__main__':
  main()