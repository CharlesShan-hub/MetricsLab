# MetricsLab

## 文件目录

* metrics：所有的指标与指标相关的loss，所有的输入都是$$(B,C,H,W)$$形状的张量
  * \_\_init\_\_：用于集合所有的指标与loss，方便导入
  * **ssim**：（VIFB，Structural similarly-based）Structural Similarity index measure
  * **rmse**：（VIFB，Structural similarly-based）Root mean squared error
  * **ce**：（VIFB，Information theory-based）Cross entropy
  * **en**：（VIFB，Information theory-based）Entropy
  * **mi**：（VIFB，Information theory-based）Mutural information
  * **psnr**：（VIFB，Information theory-based）Peak signal-to-noise ration
  * **ag**：（VIFB，Image feature-based）Average gradient
  * **ei**：（VIFB，Image feature-based）Edge intensity
  * **sd**：（VIFB，Image feature-based）Standard deviation
  * **sf**：（VIFB，Image feature-based）Spatial frequency
  * **$$Q^{AB/F}$$**：（VIFB，Image feature-based）Gradient-based fusion performance
  * **$$Q_{CB}$$**：（VIFB，Human perception inspired）Chen-Blum metric
  * **$$Q_{CV}$$**：（VIFB，Human perception inspired）Chen-Varsheny metric
* imgs：所有的图片
  * RoadScene、TNO：数据集
    * ir：红外图片
    * vis：可见光图片
    * fuse：融合图片
      * U2Fusion等融合方式

## Demo

1. 读入图片

   ```python
   from utils import *
   
   # 通过路径直接导入
   ir_tensor = read_grey_tensor('./imgs/TNO/ir/1.bmp',requires_grad=False)
   vis_tensor = read_grey_tensor('./imgs/TNO/vis/1.bmp',requires_grad=False)
   fuse_tensor = read_grey_tensor('./imgs/TNO/fuse/U2Fusion/1.bmp',requires_grad=True)
   
   # 通过信息间接导入
   ir_tensor = read_grey_tensor(dataset='TNO',category='ir',name='1.bmp',requires_grad=False)
   vis_tensor = read_grey_tensor(dataset='TNO',category='vis',name='1.bmp',requires_grad=False)
   fuse_tensor = read_grey_tensor(dataset='TNO',category='fuse',name='1.bmp',model='U2Fusion',requires_grad=True)
   ```

   