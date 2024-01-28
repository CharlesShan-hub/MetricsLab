from utils import *

# 通过路径直接导入图片
ir_tensor = read_grey_tensor('./imgs/TNO/ir/1.bmp',requires_grad=False)
vis_tensor = read_grey_tensor('./imgs/TNO/vis/1.bmp',requires_grad=False)
fuse_tensor = read_grey_tensor('./imgs/TNO/fuse/U2Fusion/1.bmp',requires_grad=True)

# 通过信息间接导入图片
ir_tensor = read_grey_tensor(dataset='TNO',category='ir',name='1.bmp',requires_grad=False)
vis_tensor = read_grey_tensor(dataset='TNO',category='vis',name='1.bmp',requires_grad=False)
fuse_tensor = read_grey_tensor(dataset='TNO',category='fuse',name='1.bmp',model='U2Fusion',requires_grad=True)


