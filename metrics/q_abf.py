import torch
import kornia

###########################################################################################

__all__ = [
    'q_abf',
    'q_abf_approach_loss',
    'q_abf_metric'
]

def q_abf(imgA, imgB, imgF, border_type='constant', eps=1e-10):
    """
    Calculate the Q_ABF (Quality Assessment for image Blurred and Fused) metric.

    Args:
        imgA (torch.Tensor): The first input image tensor.
        imgB (torch.Tensor): The second input image tensor.
        imgF (torch.Tensor): The fused image tensor.
        border_type (str, optional): Type of border extension. Default is 'constant' for adapt VIFB, but
                in kornia border_type default is 'reflection'
        eps (float, optional): A small value to avoid numerical instability. Default is 1e-10.

    Returns:
        torch.Tensor: The Q_ABF metric value.
    """
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

    (gA,aA) = edge_strength_and_orientation(imgA*255.0)
    (gB,aB) = edge_strength_and_orientation(imgB*255.0)
    (gF,aF) = edge_strength_and_orientation(imgF*255.0)
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
    #print("3. Relative Strength and Orientation")
    #print(torch.mean(GAF),torch.mean(AAF),torch.mean(GBF),torch.mean(ABF))

    # 边缘强度和方向保留值
    def edge_strength_and_orientation_preservation_values(G_F,A_F):
        Qg = Tg / (1 + torch.exp(kg * (G_F - Dg)))
        Qa = Ta / (1 + torch.exp(ka * (A_F - Da)))
        # Qg = Tg / (1 + torch.exp(torch.clamp(kg * (G_F - Dg), min=-20, max=20))) # Improve
        # Qa = Ta / (1 + torch.exp(torch.clamp(ka * (A_F - Da), min=-20, max=20))) # Improve
        return Qg * Qa

    QAF = edge_strength_and_orientation_preservation_values(GAF,AAF)
    QBF = edge_strength_and_orientation_preservation_values(GBF,ABF)
    #print("4. Edge Strength and Orientation Preservation Values")
    #print(torch.mean(QAF),torch.mean(QBF))

    deno = torch.sum(gA + gB)
    nume = torch.sum(QAF * gA + QBF * gB)
    qabf_value = nume / deno

    return qabf_value

# 采用相同图片的 q_abf 减去不同图片的 q_abf
def q_abf_approach_loss(A, F):
    # return q_abf(A, A, A)-q_abf(A, A, F)
    # return 1-q_abf(A, A, F)
    return 0.9748 - q_abf(A, A, F)

# 与 VIFB 统一
def q_abf_metric(A, B, F):
    return q_abf(A, B, F)

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

    print(f'Q_ABF(ir,ir,ir):{q_abf(ir,ir,ir)}')         # 0.9747936129570007
    print(f'Q_ABF(vis,vis,vis):{q_abf(vis,vis,vis)}')   # 0.9747936129570007
    print(f'Q_ABF(vis,ir,fused):{q_abf(vis,ir,fused)}') # 0.43425410985946655

if __name__ == '__main__':
    main()
