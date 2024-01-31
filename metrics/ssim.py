# kornia
# https://kornia.readthedocs.io/en/latest/metrics.html#kornia.metrics.ssim
from kornia.metrics import ssim as ssim_kornia
from kornia.metrics import ssim3d as ssim3d_kornia
from kornia.losses import ssim_loss as ssim_loss_kornia
from kornia.losses import ssim3d_loss as ssim3d_loss_kornia
from torch import mean
def ssim_metric(imgA,imgB,imgF):
    w0 = w1 = 0.5
    return mean(w0 * ssim_kornia(imgF,imgA,window_size=11) + w1 * ssim_kornia(imgF,imgB,window_size=11))

# pytorch-msssim
# https://github.com/VainF/pytorch-msssim
#from pytorch_msssim import ssim as ssim_differentiable
#from pytorch_msssim import ms_ssim as ms_ssim_differentiable