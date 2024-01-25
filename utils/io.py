import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.autograd import Variable

def numpy_to_tensor(image):
  transform = transforms.Compose([transforms.ToTensor()])
  return transform(image).unsqueeze(0)

def grey_tensor_to_image(tensor):
  image = tensor.squeeze().detach().numpy()
  image = np.clip(image * 255, 0, 255).astype(np.uint8)
  return image

def read_grey_tensor(path=None,requires_grad=True,dataset=None,category=None,name=None,model=None,base_path='./imgs'):
  if path == None:
    if model == None:
      path = os.path.join(base_path,dataset,category,name)
    else:
      path = os.path.join(base_path,dataset,category,model,name)
  img_grey = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  return Variable(numpy_to_tensor(img_grey).float(), requires_grad=requires_grad)

def read_rgb_tensor(path=None,requires_grad=True,dataset=None,category=None,name=None,model=None,base_path='./imgs'):
  if path == None:
    if model == None:
      path = os.path.join(base_path,dataset,category,name)
    else:
      path = os.path.join(base_path,dataset,category,model,name)
  img = cv2.imread(path, cv2.IMREAD_COLOR)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return Variable(numpy_to_tensor(img_rgb).float(), requires_grad=requires_grad)

def save_grey_image(folder_name,image,epoch,base_path='./logs'):
  base_path = os.path.join(base_path,folder_name,'imgs')
  if not os.path.exists(base_path):
    os.makedirs(base_path)
  cv2.imwrite(f"{os.path.join(base_path,str(epoch))}.png", image)

def save_grey_images(folder_name,images,base_path='./logs'):
  for (epoch,image) in enumerate(images):
    save_grey_image(folder_name,image,epoch)

def video_grey_images(folder_name,images,base_path='./logs'):
  (height, width) = images[0].shape
  video = cv2.VideoWriter(f"{os.path.join(base_path,folder_name,'Progress')}.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))
  for image in images:
    video.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
  video.release()
