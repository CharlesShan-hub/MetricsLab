# 本案例用于验证指标与 VIFB 的一致性
import csv
from metrics.eva import eva_metric
from utils import *

from metrics import ce_metric
from metrics import en_metric
from metrics import mi_metric
from metrics import psnr_metric
from metrics import ssim_metric
from metrics import rmse_metric
from metrics import ag_metric
from metrics import ei_metric
from metrics import sd_metric
from metrics import sf_metric
from metrics import q_abf_metric
from metrics import q_cb_metric
from metrics import q_cv_metric

from metrics import cc_metric
from metrics import scd_metric
from metrics import snr_metric
from metrics import te_metric
from metrics import nmi_metric
from metrics import q_ncie_metric
from metrics import q_w_metric
from metrics import q_c_metric
from metrics import q_y_metric
from metrics import eva_metric
from metrics import asm_metric
from metrics import sam_metric
from metrics import mse_metric
from metrics import vif_metric
from metrics import con_metric
from metrics import nrmse_metric
from metrics import fmi_w_metric
from metrics import fmi_g_metric
from metrics import fmi_d_metric
from metrics import fmi_e_metric
from metrics import fmi_p_metric
from metrics import ergas_metric
from metrics import q_p_metric
from metrics import n_abf_metric
from metrics import pww_metric

def main():
    name_list = ['U2Fusion','ADF','CBF', 'CNN', 'FPDE', 'GFCE', 'GTF', 'HMSD_GF', 'IFEVIP', 'LatLRR', 'MSVD', 'TIF', 'VSMWLS']
    # name_list = ['U2Fusion']
    ir_tensor = read_grey_tensor(dataset='TNO',category='ir',name='9.bmp',requires_grad=False)
    vis_tensor = read_grey_tensor(dataset='TNO',category='vis',name='9.bmp',requires_grad=False)
    tensor_list = []
    for name in name_list:
        tensor_list.append(read_grey_tensor(dataset='TNO',category='fuse',name='9.bmp',model=name,requires_grad=True))

    VIFB_dist = {
        # 'CE':ce_metric,              # 通过，完全一致
        # 'EN':en_metric,              # 通过，完全一致
        # 'mi':mi_metric,              # 通过，结果与 VIFB 未进行归一化计算的结果完全一致，归一化后核估计不能正常拟合
        # 'PSNR':psnr_metric,          # 通过。该代码 VIFB 有错，以本代码为准
        # 'SSIM':ssim_metric,          # 通过，kornia 官方实现。该代码 VIFB 有错（没除 2），以本代码为准
        # 'RMSE':rmse_metric,          # 通过，完全一致
        # 'AG':ag_metric,              # 通过，1e-5 级别上有误差,因为 eps=1e-10
        # 'EI':ei_metric,              # 通过，1e-5 级别上有误差,因为 eps=1e-10
        # 'SD':sd_metric,              # 通过，1e-5 级别上有误差,因为 eps=1e-10
        # 'SF':sf_metric,              # 通过，该代码 VIFB 有错，以本代码为准（VIFB求平均求错了，像素个数写成了 mn 其实是 m(n-1)）
        # 'Q_ABF':q_abf_metric,        # 通过，小数点后两位无误差。后续再调整
        # 'Q_CB':q_cb_metric,          # 通过，小数点后一位无误差。后续再调整
        # 'Q_CV':q_cv_metric,          # 通过, 完全一致（VIFB）
    }

    metric_dist = {
        # 'CC':cc_metric,              # 通过，完全一致（Tang）
        # 'SCD':scd_metric,            # 通过，完全一致（Tang）
        # 'SNR':snr_metric,            # 可以运行
        # 'TE':te_metric,              # 后续修改，大部分与 MEFB 非均匀化的结果完全一致，少部分有较大区别
        # 'NMI':nmi_metric,            # 通过，结果与 MEFB 未进行归一化计算的结果完全一致，归一化后核估计不能正常拟合
        # 'Q_NCIE':q_ncie_metric,      # 通过，结果与 MEFB 未进行归一化计算的结果完全一致，归一化后核估计不能正常拟合
        # 'Q_W':q_w_metric,            # 通过，完全一致（MEFB）
        # 'Q_C':q_c_metric,            # 通过，与 MEFB 几乎一致。后续再调整
        # 'Q_Y':q_y_metric,            # 通过，完全一致（MEFB）
        # 'EVA':eva_metric,            # 通过，大概结果的区间与论文一致，博主实现的有错误，按此版本为准
        # 'ASM':asm_metric,            # 不可微!!!
        # 'SAM':sam_metric,            # 非常不常用??
        # 'MSE':mse_metric,            # 通过
        # 'CON':con_metric,            # 通过
        # 'NRMSE':nrmse_metric,        # 通过
        # 'FMI(pixel)':fmi_p_metric,   # 通过，已经改成了卷积形式
        # 'FMI(edge)':fmi_e_metric,    # 通过，已经改成了卷积形式
        # 'FMI(dct)':fmi_d_metric,     # 通过，已经改成了卷积形式
        # 'FMI(gradient)':fmi_g_metric,# 通过，已经改成了卷积形式
        # 'FMI(wavelet)':fmi_w_metric, # 通过，已经改成了卷积形式
        # N_ABF: 'n_abf_metrics',      # 通过，完全一致（Many）
        # PWW: 'pww_metrics',          # 通过，完全一致（Many）
        # 'VIF': vif_metric,           # 通过，完全一致（Many）
        # 'VIFF': viff_metric,         #
        # 'ERGAS': ergas_metric,       # 通过
        # 'Q_P': q_p_metric,           #
    }
    metrics_data = []
    metrics_data.append(['Metric Demo']+name_list)
    for metric in metric_dist:
        metric_data = []
        for (name,fuse_tensor) in zip(name_list,tensor_list):
            value = metric_dist[metric](vis_tensor,ir_tensor,fuse_tensor).item()
            print(metric,name,':',value)
            metric_data.append(value)
        metrics_data.append([metric]+metric_data)
        print(metric_data)
    with open('./logs/metrics_demo.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(metrics_data)

if __name__ == '__main__':
  main()
