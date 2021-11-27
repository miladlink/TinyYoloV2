import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import numpy as np

class MaxPoolStride1 (nn.Module):
    """ maxpooling without changing size """
    def __init__ (self):
        super (MaxPoolStride1, self).__init__()
   
    def forward (self, x):
        x = F.max_pool2d (F.pad (x, (0, 1, 0, 1), mode = 'replicate'), 2, stride = 1)
        return x
    
    
def load_conv_bn (buf, start, conv_model, bn_model):
    """ fitting the (.weight) weights to the BatchNormalization & Conv layer """
    # number of conv & bn weights (in yolo conv layer has no bias except last layer)
    num_w = conv_model.weight.numel ()
    num_b = bn_model.bias.numel ()
    
    # place weights to the bn & conv layer respectively
    bn_model.bias.data.copy_ (torch.from_numpy (buf [start: start + num_b])); start = start + num_b
    bn_model.weight.data.copy_ (torch.from_numpy (buf [start: start + num_b])); start = start + num_b
    bn_model.running_mean.copy_ (torch.from_numpy (buf [start: start + num_b])); start = start + num_b
    bn_model.running_var.copy_ (torch.from_numpy (buf [start: start + num_b])); start = start + num_b
    
    conv_model.weight.data.copy_ (torch.from_numpy (buf [start: start + num_w]).reshape_as (conv_model.weight)); start = start + num_w
    return start


def load_conv (buf, start, conv_model):
    """ fitting the (.weight) weights to last conv layer """
    # numer of weights & bias
    num_w = conv_model.weight.numel ()
    num_b = conv_model.bias.numel ()
    # place weights & bias to conv layer
    conv_model.bias.data.copy_ (torch.from_numpy (buf [start: start + num_b])); start = start + num_b
    conv_model.weight.data.copy_ (torch.from_numpy (buf [start: start + num_w]).reshape_as (conv_model.weight)); start = start + num_w
    
    
class TinyYoloV2 (nn.Module):
    def __init__ (self):
        super (TinyYoloV2, self).__init__()
        self.seen = 0
        self.num_classes = 20 # VOC dataset
        self.anchors = [1.08,1.19, 3.42,4.41, 6.63,11.38, 9.42,5.11, 16.62,10.52] # each pair is a anchor
        self.num_anchors = len (self.anchors) / 2
        num_output = (5 + self.num_classes) * self.num_anchors 
        self.width = 160 # 416
        self.height = 160 # 416
        self.cnn = nn.Sequential (OrderedDict ([
            ('conv1' ,  nn.Conv2d (3, 16, 3, 1, 1, bias = False)), 
            ('bn1'   ,  nn.BatchNorm2d (16)),
            ('leaky1',  nn.LeakyReLU (0.1, inplace = True)),
            ('pool1' ,  nn.MaxPool2d (2, 2)),                        # (16, 208, 208)
            
            ('conv2' ,  nn.Conv2d (16, 32, 3, 1, 1, bias = False)),
            ('bn2'   ,  nn.BatchNorm2d (32)),
            ('leaky2',  nn.LeakyReLU (0.1, inplace = True)),
            ('pool2' ,  nn.MaxPool2d (2, 2)),                         # (32, 104, 104)
            
            ('conv3' ,  nn.Conv2d (32, 64, 3, 1, 1, bias = False)),
            ('bn3'   ,  nn.BatchNorm2d (64)),
            ('leaky3',  nn.LeakyReLU (0.1, inplace = True)),
            ('pool3' ,  nn.MaxPool2d (2, 2)),                         # (64, 52, 52)
            
            ('conv4' ,  nn.Conv2d (64, 128, 3, 1, 1, bias = False)),
            ('bn4'   ,  nn.BatchNorm2d (128)),
            ('leaky4',  nn.LeakyReLU (0.1, inplace = True)),
            ('pool4' ,  nn.MaxPool2d (2, 2)),                         # (128, 26, 26)
            
            ('conv5' ,  nn.Conv2d (128, 256, 3, 1, 1, bias = False)),
            ('bn5'   ,  nn.BatchNorm2d (256)),
            ('leaky5',  nn.LeakyReLU (0.1, inplace = True)),
            ('pool5' ,  nn.MaxPool2d (2, 2)),                         # (256, 13, 13)
            
            ('conv6' ,  nn.Conv2d (256, 512, 3, 1, 1, bias = False)),
            ('bn6'   ,  nn.BatchNorm2d (512)),
            ('leaky6',  nn.LeakyReLU (0.1, inplace = True)),
            ('pool6' ,  MaxPoolStride1()),                            # (512, 13, 13)
            
            ('conv7' ,  nn.Conv2d (512, 1024, 3, 1, 1, bias = False)),
            ('bn7'   ,  nn.BatchNorm2d (1024)),
            ('leaky7',  nn.LeakyReLU (0.1, inplace = True)),          # (1024, 13, 13)
            
            ('conv8' ,  nn.Conv2d (1024, 1024, 3, 1, 1, bias = False)),
            ('bn8'   ,  nn.BatchNorm2d (1024)),
            ('leaky8',  nn.LeakyReLU (0.1, inplace = True)),          # (1024, 13, 13)
            # (5 + num_classes) * num_ancors = (5 + 20) * 5 = 125
            ('output',  nn.Conv2d (1024, 125, 1, 1, 0))               # (125, 13, 13)
          ]))
  
    def forward (self, x):
        x = self.cnn (x)
        return x
  
    def print_network (self):
        print (self)
  
    def load_weights (self, path):
        # read .weight => returns list of numpy
        buf = np.fromfile (path, dtype = np.float32)
        start = 4 # 4 first numbers does not contain weights
          
        # copy weights to per layer
        # start = load_conv_bn (buf, start, conv_model, bn_model)
        start = load_conv_bn (buf, start, self.cnn [0], self.cnn [1])
        start = load_conv_bn (buf, start, self.cnn [4], self.cnn [5])
        start = load_conv_bn (buf, start, self.cnn [8], self.cnn [9])
        start = load_conv_bn (buf, start, self.cnn [12], self.cnn [13])
        start = load_conv_bn (buf, start, self.cnn [16], self.cnn [17])
        start = load_conv_bn (buf, start, self.cnn [20], self.cnn [21])
        # after previous layers maxpooling removed
        start = load_conv_bn (buf, start, self.cnn [24], self.cnn [25])
        start = load_conv_bn (buf, start, self.cnn [27], self.cnn [28])
        # just conv layer
        start = load_conv (buf, start, self.cnn [30])    
    