from metrics import en_metric
from metrics import mi_metric
from metrics import psnr_metric
from metrics import q_abf_metric
from metrics import sd_metric
from metrics import sf_metric
from metrics import ssim_metric
from metrics import vif_metric
from metrics import scd_metric
from metrics import cc_metric

metirc_dict = {
    'en': en_metric,
    'mi': mi_metric,
    'psnr': psnr_metric,
    'q_abf': q_abf_metric,
    'sd': sd_metric,
    'sf': sf_metric,
    'ssim': ssim_metric,
    'vif': vif_metric,
    'scd': scd_metric,
    'cc': cc_metric
}

import os
def get_all_files_in_path(path):
    files = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            files.append(file)
    return files
path = './metrics'
files = get_all_files_in_path(path)

for file in files:
    metrics_data
