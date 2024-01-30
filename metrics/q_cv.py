import torch
import torch.fft
import kornia
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

def q_cv(imgA, imgB, imgF, border_type='constant'):
    return

def q_cv_loss(imgA, imgB, imgF):
  return -q_cv(imgA, imgB, imgF)


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

    print(q_cv(img1,img2,fused))
    print(q_cv(img1,img1,img1))

if __name__ == '__main__':
  main()

