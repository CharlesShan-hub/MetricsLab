# 本案例用于验证指标与 VIFB 的一致性

from metrics import ce_loss
from metrics import en_loss
from metrics import mi_loss
from metrics import psnr_loss
from metrics import ssim_loss
from metrics import rmse_loss
from metrics import ag_loss
from metrics import ei_loss
from metrics import sd_loss
from metrics import sf_loss
from metrics import q_abf
from metrics import q_cb

def main():
    ir_tensor = read_grey_tensor(dataset='TNO',category='ir',name='9.bmp',requires_grad=False)
    vis_tensor = read_grey_tensor(dataset='TNO',category='vis',name='9.bmp',requires_grad=False)
    fuse_tensor1 = read_grey_tensor(dataset='TNO',category='fuse',name='9.bmp',model='U2Fusion',requires_grad=True)
    fuse_tensor2 = read_grey_tensor(dataset='TNO',category='fuse',name='9.bmp',model='FPDE',requires_grad=True)
    fuse_tensor3 = read_grey_tensor(dataset='TNO',category='fuse',name='9.bmp',model='ADF',requires_grad=True)
    fuse_tensor4 = read_grey_tensor(dataset='TNO',category='fuse',name='9.bmp',model='Average',requires_grad=True)
    fuse_tensor5 = read_grey_tensor(dataset='TNO',category='fuse',name='9.bmp',model='CBF',requires_grad=True)
    fuse_tensor6 = read_grey_tensor(dataset='TNO',category='fuse',name='9.bmp',model='CDDFuse',requires_grad=True)
    fuse_tensor7 = read_grey_tensor(dataset='TNO',category='fuse',name='9.bmp',model='FusionGAN',requires_grad=True)
    fuse_tensor8 = read_grey_tensor(dataset='TNO',category='fuse',name='9.bmp',model='GFCE',requires_grad=True)

    tensor_list = [fuse_tensor1,fuse_tensor2,fuse_tensor3,fuse_tensor4,fuse_tensor5,fuse_tensor6,fuse_tensor7,fuse_tensor8]
    name_list = ['U2Fusion','FPDE','ADF'，'Average', 'CBF', 'CDDFuse', 'FusionGAN', 'GFCE']
    metric_list = [
        ce_loss
    ]

    for metric in metric_list:
        for (name,fuse_tensor) in enumerate(name_list,tensor_list):
            pass


if __name__ == '__main__':
  main()