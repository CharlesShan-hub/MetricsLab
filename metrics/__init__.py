'''
 * 方法(): 默认输入都是 0-1 的张量
 * 方法_metric(): 默认输入都是 0-1 的张量, 但会调整调用方法()的输入，会与 VIFB 一致
 * 方法_approach_loss()：默认输入都是 0-1 的张量，用于趋近测试
 *
'''


# 信息论
from metrics.ce import *       # VIFB - 交叉熵
from metrics.en import *       # VIFB - 信息熵
from metrics.mi import *       # VIFB - 互信息
from metrics.snr import *      # Many - 信噪比
from metrics.psnr import *     # VIFB - 峰值信噪比

# 结构相似性
from metrics.ssim import *     # VIFB - 结构相似度测量
from metrics.ms_ssim import *  # Tang - 多尺度结构相似度测量
from metrics.rmse import *     # VIFB - 均方误差
# from metrics.ergas import *    # Many
# from metrics.sam import *      # Many

# 图片信息
from metrics.ag import *       # VIFB - 平均梯度
from metrics.ei import *       # VIFB - 边缘强度
# from metrics.mb import *       # Many
# from metrics.pfe import *      # Many
from metrics.sd import *       # VIFB - 标准差
from metrics.sf import *       # VIFB - 空间频率
from metrics.q_abf import *    # VIFB - 基于梯度的融合性能

# 视觉感知
from metrics.q_cb import *     # VIFB - 图像模糊与融合的质量评估
# from metrics.vif import *      # Tang - 视觉保真度

# 新指标暂时没分类
from metrics.cc import *       # Tang - 相关系数
from metrics.scd import *      # Tang - 差异相关和
# from metrics.n_abf import *    # Tang - 基于噪声评估的融合性能

# from metrics.fmi_w import *    #
# from metrics.fmi_dct import *  #
# from metrics.fmi_pixel import *#
# from metrics.df import *       #
# from metrics.q_sf import *     #
# from metrics.q_mi import *     #
# from metrics.q_s import *      #
# from metrics.q_y import *      #
# from metrics.q_c import *      #
# from metrics.q_ncie import *   #
# from metrics.mi_abf import *   #
# from metrics.viff import *     #
# from metrics.q_p import *      #
# from metrics.q_w import *      #
# from metrics.q_e import *      #
# from metrics.uqi import *      # Many
# from metrics.qi import *       # Many
# from metrics.theta import *    # Many
# from metrics.fqi import *      # Many
# from metrics.fsm import *      # Many
# from metrics.wfqi import *     # Many
# from metrics.efqi import *     # Many
# from metrics.d import *        # Many
