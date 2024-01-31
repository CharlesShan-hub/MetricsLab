# 本案例用于验证指标与 VIFB 的一致性
import csv
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

def main():
    name_list = ['U2Fusion','ADF','CBF', 'CNN', 'FPDE', 'GFCE', 'GTF', 'HMSD_GF', 'IFEVIP', 'LatLRR', 'MSVD', 'TIF', 'VSMWLS']
    ir_tensor = read_grey_tensor(dataset='TNO',category='ir',name='9.bmp',requires_grad=False)
    vis_tensor = read_grey_tensor(dataset='TNO',category='vis',name='9.bmp',requires_grad=False)
    tensor_list = []
    for name in name_list:
        tensor_list.append(read_grey_tensor(dataset='TNO',category='fuse',name='9.bmp',model=name,requires_grad=True))
    
    metric_dist = {
        #'CE':ce_metric,       #
        #'EC':en_metric,       #
        #'mi':mi_metric,       #
        'PSNR':psnr_metric,   # 通过，
        #'SSIM':ssim_metric,   #
        #'RMSE':rmse_metric,   #
        #'AG':ag_metric,       # 通过，1e-5 级别上有误差,因为 eps=1e-10
        #'EI':ei_metric,       # 通过，1e-5 级别上有误差,因为 eps=1e-10
        #'SD':sd_metric,       # 通过，1e-5 级别上有误差,因为 eps=1e-10
        #'SF':sf_metric,       # 通过，小数点后一位无误差。后续再调整
        #'Q_ABF':q_abf_metric, #
        #'Q_CB':q_cb_metric    #
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
