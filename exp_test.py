import torch
import kornia
from utils import *
from metrics import fmi_w_metric
import time

A = read_grey_tensor(dataset='TNO',category='ir',name='9.bmp',requires_grad=False)
B = read_grey_tensor(dataset='TNO',category='vis',name='9.bmp',requires_grad=False)
F1 = read_grey_tensor(dataset='TNO',category='fuse',name='9.bmp',model='U2Fusion',requires_grad=True)
F = read_grey_tensor(dataset='TNO',category='fuse',name='9.bmp',model='U2Fusion',requires_grad=True)

def toy_method(method):
    start_time = time.time()
    method()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time of {method.__name__}: {execution_time} seconds")

def method_1():
    sameAF = kornia.filters.filter2d(A-F,torch.tensor([[[1,1,1],[1,1,1],[1,1,1]]]),padding='valid')
    sameAF = torch.where(sameAF == 0, torch.tensor(0.0), torch.tensor(1.0))

_, _, m, n = A.shape
w = 1
fmi_map = torch.ones((m - 2 * w, n - 2 * w))
def method_2():
    for p in range(w, m - w):
        for q in range(w, n - w):
            a_sub = A[:,:,p - w:p + w + 1, q - w:q + w + 1]
            f_sub = F[:,:,p - w:p + w + 1, q - w:q + w + 1]
            if torch.equal(a_sub, f_sub):
                fmi_map[p-w,q-w] = torch.tensor(1.0)
            else:
                fmi_map[p-w,q-w] = torch.tensor(0.0)


# 测试方法 1 的执行时间
toy_method(method_1)

# 测试方法 2 的执行时间
toy_method(method_2)
