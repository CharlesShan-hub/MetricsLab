# 本实验用于测试各种指标是否运行正确

from utils import *

ir_tensor = read_grey_tensor('./imgs/TNO/ir/1.bmp',requires_grad=False)
vis_tensor = read_grey_tensor('./imgs/TNO/vis/1.bmp',requires_grad=False)
fuse_tensor = read_grey_tensor('./imgs/TNO/fuse/U2Fusion/1.bmp',requires_grad=True)

ir_tensor = read_grey_tensor(dataset='TNO',category='ir',name='1.bmp',requires_grad=False)
vis_tensor = read_grey_tensor(dataset='TNO',category='vis',name='1.bmp',requires_grad=False)
fuse_tensor = read_grey_tensor(dataset='TNO',category='fuse',name='1.bmp',model='U2Fusion',requires_grad=True)

