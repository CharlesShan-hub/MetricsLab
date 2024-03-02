import torch
import kornia

###########################################################################################

__all__ = [
    'q_sf',
    'q_sf_approach_loss',
    'q_sf_metric'
]

def q_sf(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor,
    border_type: str = 'replicate', eps: float = 1e-10) -> torch.Tensor:
    """
    Calculates the Q_SF metric between two images A and B
    with respect to a fused image F.

    Args:
        A (torch.Tensor): The first input image tensor.
        B (torch.Tensor): The second input image tensor.
        F (torch.Tensor): The fused image tensor.
        border_type (str, optional): The padding mode for convolution. Default is 'replicate'.
        eps (float, optional): Small value to prevent division by zero. Default is 1e-10.

    Returns:
        torch.Tensor: The SF quality metric value.
    """

    def calculate_grad(I):
        RF = kornia.filters.filter2d(I,torch.tensor([[[1],[-1]]]),border_type=border_type)
        CF = kornia.filters.filter2d(I,torch.tensor([[[ 1, -1]]]),border_type=border_type)
        MDF= kornia.filters.filter2d(I,torch.tensor([[[-1,0],[0,1]]]),border_type=border_type)
        SDF= kornia.filters.filter2d(I,torch.tensor([[[0,-1],[1,0]]]),border_type=border_type)
        [MDF,SDF] = [G/torch.sqrt(torch.tensor(2.0)) for G in [MDF,SDF]]
        [RF,CF,MDF,SDF] = [torch.abs(G) for G in [RF,CF,MDF,SDF]]
        return (RF,CF,MDF,SDF)

    def calculate_sf(RF,CF,MDF,SDF):
        [RF,CF,MDF,SDF] = [torch.mean(G**2) for G in [RF,CF,MDF,SDF]]
        return torch.sqrt(RF+CF+MDF+SDF+eps)

    [RFF,CFF,MDFF,SDFF] = calculate_grad(F)
    [RFR,CFR,MDFR,SDFR] = [torch.max(GA,GB) for (GA,GB) in zip(calculate_grad(A),calculate_grad(B))]
    # import matplotlib.pyplot as plt
    # plt.subplot(2,2,1)
    # plt.imshow(RFR.squeeze().detach().numpy())
    # plt.subplot(2,2,2)
    # plt.imshow(CFR.squeeze().detach().numpy())
    # plt.subplot(2,2,3)
    # plt.imshow(MDFR.squeeze().detach().numpy())
    # plt.subplot(2,2,4)
    # plt.imshow(SDFR.squeeze().detach().numpy())
    # plt.show()
    SFF = calculate_sf(RFF,CFF,MDFF,SDFF)
    SFR = calculate_sf(RFR,CFR,MDFR,SDFR)
    return (SFF-SFR)/SFR

def q_sf_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return 1 - q_sf(A,A,F)

def q_sf_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return q_sf(A*255, B*255, F*255)

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

    print(f'Q_SF(vis,ir,fused):{q_sf_metric(vis,ir,fused)}')
    print(f'Q_SF(vis,vis,vis):{q_sf_metric(vis,vis,vis)}')
    print(f'Q_SF(vis,vis,fused):{q_sf_metric(vis,vis,fused)}')
    print(f'Q_SF(vis,vis,ir):{q_sf_metric(vis,vis,ir)}')

if __name__ == '__main__':
    main()
