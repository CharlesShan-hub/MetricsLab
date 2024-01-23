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