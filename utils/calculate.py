import matplotlib.pyplot as plt
import os
import numpy as np

def plot_and_save_loss_with_lr(folder_name, loss_list, learning_rate, save_path='loss_plot.png', txt_path='loss_values.txt',base_path='./logs'):
  base_path = os.path.join(base_path,folder_name)
  # 创建 x 轴的轮次
  epochs = list(range(1, len(loss_list) + 1))

  # 绘制 loss 曲线
  plt.plot(epochs, loss_list, label='Training Loss')
  plt.title(f'Training Loss Over Epochs (Learning Rate: {learning_rate})')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

  # 保存 loss 曲线图
  plt.savefig(os.path.join(base_path,save_path))
  plt.close()

  # 保存 loss 到 txt 文件
  np.savetxt(os.path.join(base_path,txt_path), np.array(loss_list), delimiter='\n')

  