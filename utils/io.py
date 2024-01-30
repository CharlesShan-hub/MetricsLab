import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

def numpy_to_tensor(image):
  transform = transforms.Compose([transforms.ToTensor()])
  return transform(image).double().unsqueeze(0)

def grey_tensor_to_image(tensor):
  image = tensor.squeeze().detach().numpy()
  image = np.nan_to_num(image, nan=0.0)
  image = np.clip(image * 255, 0, 255).astype(np.uint8)
  return image

def read_grey_tensor(path=None,requires_grad=True,dataset=None,category=None,name=None,model=None,base_path='./imgs'):
  if path == None:
    if model == None:
      path = os.path.join(base_path,dataset,category,name)
    else:
      path = os.path.join(base_path,dataset,category,model,name)
  img_grey = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  return Variable(numpy_to_tensor(img_grey), requires_grad=requires_grad)

def read_rgb_tensor(path=None,requires_grad=True,dataset=None,category=None,name=None,model=None,base_path='./imgs'):
  if path == None:
    if model == None:
      path = os.path.join(base_path,dataset,category,name)
    else:
      path = os.path.join(base_path,dataset,category,model,name)
  img = cv2.imread(path, cv2.IMREAD_COLOR)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return Variable(numpy_to_tensor(img_rgb).float(), requires_grad=requires_grad)

def save_grey_image(folder_name,image,epoch,name='',base_path='./logs'):
  base_path = os.path.join(base_path,folder_name,'imgs')
  if not os.path.exists(base_path):
    os.makedirs(base_path)
  cv2.imwrite(f"{os.path.join(base_path,name+str(epoch))}.png", image)

def save_grey_images(folder_name,images,step=1,first_ten=True,name='',base_path='./logs'):
  for (epoch,image) in enumerate(images):
    if epoch%step==0:
      save_grey_image(folder_name,image,epoch,name)
    elif first_ten==True and epoch <= 10:
      save_grey_image(folder_name,image,epoch,name)

def video_grey_images(folder_name,images,name='Progress',base_path='./logs'):
  (height, width) = images[0].shape
  video = cv2.VideoWriter(f"{os.path.join(base_path,folder_name,name)}.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))
  for image in images:
    video.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
  video.release()

def save_result(folder_name,vis_result,ir_result,vis,ir,base_path='./logs'):
  vis = grey_tensor_to_image(vis)
  ir = grey_tensor_to_image(ir)
  plt.subplot(2,3,1)
  plt.imshow(vis_result,cmap='gray')
  plt.title("VIS Result")
  plt.subplot(2,3,2)
  plt.imshow(vis,cmap='gray')
  plt.title("VIS")
  plt.subplot(2,3,3)
  plt.imshow(np.abs(vis-vis_result),cmap='gray')
  plt.title("VIS Difference")
  plt.subplot(2,3,4)
  plt.imshow(ir_result,cmap='gray')
  plt.title("IR Result")
  plt.subplot(2,3,5)
  plt.imshow(ir,cmap='gray')
  plt.title("IR")
  plt.subplot(2,3,6)
  plt.imshow(np.abs(ir-ir_result),cmap='gray')
  plt.title("IR Difference")
  plt.savefig(os.path.join(base_path,folder_name,'result.png'))
  plt.close()
