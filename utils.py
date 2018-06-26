import copy
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt


def show_heatmap(X):
    x = copy.deepcopy(X)
    cmap = plt.cm.get_cmap("seismic")
    shape = x.shape
    x = x.sum(axis=0)
    x /= np.max(np.abs(x)) # -1,1
    x += 1. # 0, 2
    x *= 127.5 # 0, 255
    x = x.astype(np.int64)
    x_cmap = cmap(x.flatten())[:,:3].T
    return x_cmap.reshape(shape)


def show_bp(X):
    x = copy.deepcopy(X)
    x /= np.max(np.abs(x)) # (-1, 1)
    x /= 2. # (-0.5, 0.5)
    x += 0.5 # (0, 1)
    return x


def Path_to_Torch4d(image_path):
    im = Image.open(image_path)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])
    transform_plt = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    image = transform(im)  # (c, h, w), torch.Tensor
    plt_img = transform_plt(im)
    image = image.unsqueeze(0) # (1, c, h, w)
    return image, plt_img


def one_hot(output, target=None):
    if target is not None:
        target_idx = target
    else:
        target_idx = output.data.max(1)[1]
    out_shape=output.shape
    one_hot=torch.zeros(out_shape).cuda()
    one_hot[:, int(target_idx)] = 1
    one_hot = Variable(one_hot * output, requires_grad=True) # (1, 1000)
    return one_hot
    