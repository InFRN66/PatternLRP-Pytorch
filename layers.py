import torch
import torchvision
from torchvision import models, transforms, datasets
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn


class Linear(nn.Linear):
    def __init__(self, linear):
        super(nn.Linear, self).__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        """
        Linearはそこが親クラス，Linear自体はnn.Moduleを継承しているのでinit()内に引数はいらない．
        つまりLinearはそのクラス(Linear)でinitが完結してる？
        一方ConvとかMaxPoolはその上に_ConvNdとか_MaxPoolNdクラスがあるので，それら上のクラスに情報を渡すために
        init()内で引数指定している？initを_ConvNd内で行っている？
        """

    def pattern_LRP(self, R):
        B = self.weight * self.A # (out, in)
        R = R.mm(B) # (1, out) .mm (out, in) = (1, in)
        return R


class ReLU(nn.ReLU):
    #def __init__(self, ReLU):
    #    super(ReLU, self).__init__(inplace=True)
    def pattern_LRP(self, R):
        Rel_mask = (self.X > 0.)
        R = R * Rel_mask.type(torch.cuda.FloatTensor)
        return R


class Conv2d(nn.Conv2d):
    def __init__(self, conv2d):
        super(nn.Conv2d, self).__init__(conv2d.in_channels, conv2d.out_channels,
                                        conv2d.kernel_size, conv2d.stride,
                                        conv2d.padding, conv2d.dilation, 
                                        conv2d.transposed, conv2d.output_padding, 
                                        conv2d.groups, True)
        self.weight = conv2d.weight
        self.bias = conv2d.bias
        
    def backprop(self, X, B):
        return F.conv_transpose2d(input=X, weight=B, stride=self.stride, padding=self.padding)

    def pattern_LRP(self, R):
        B = self.weight * self.A
        R = self.backprop(R, B)
        return R


class MaxPool2d(nn.MaxPool2d):
    def __init__(self, maxpool2d):
        super(nn.MaxPool2d, self).__init__(maxpool2d.kernel_size,
                                           maxpool2d.stride,
                                           maxpool2d.padding,
                                           maxpool2d.dilation,
                                           maxpool2d.return_indices,
                                           maxpool2d.ceil_mode)

    def backprop(self, Z):
        temp, indices = F.max_pool2d(self.X, kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding, dilation=self.dilation, ceil_mode=self.ceil_mode,
                                     return_indices=True)  # (require : return_indices=True)
        return F.max_unpool2d(Z, indices, self.kernel_size, self.stride, self.padding)

    def pattern_LRP(self, R):
        Z = R / (self.Y + 1e-9)
        S = self.backprop(Z)
        R = self.X * S
        return R


class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, X):
        self.in_shape = X.shape
        return X.view(self.in_shape[0], -1)

    def pattern_LRP(self, R):
        return R.view(self.in_shape)
