# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:01:59 2021

@author: Jayanth
"""

import torch
from torch import nn
from utils import *

import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGBase(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """
    
    def __init__(self):
        super(VGGBase,self).__init__()
        
        # standard convolution layers in VGG16
        self.conv1_1 = nn.Conv2d(3,64,kernel_size=3,padding=3) # stride=1 by default
        self.conv1_2 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv2_1 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv2_2 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv3_1 = nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.conv3_2 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.conv3_3 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        
        self.conv4_1 = nn.Conv2d(256,512,kernel_size=3,padding=1)
        self.conv4_2 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.conv4_3 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)
        
        #Replacements of FC6 and FC7 in VGG16
        self.conv6 = nn.Conv2d(512,1024,kernel_size=3,padding=6,dilation=6)
        self.conv7 = nn.Conv2d(1024,1024,kernel_size=1)
        
        # Loading pretrained layers:
        self.load_pretrained_layers()
        
    
    def forward(self,image):
        """
        Forward propagation.
        
        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7
        """
        
        out = F.relu(self.conv1_1(image)) # (N, 64, 300, 300)
        out = F.relu(self.conv1_2(out)) # (N, 64, 300, 300)
        out = self.pool1(out) # (N, 64, 150, 150)
        
        out = F.relu(self.conv2_1(out)) # (N, 128, 150, 150)
        out = F.relu(self.conv2_2(out)) # (N, 128, 150, 150)
        out = self.pool2(out) # (N, 128, 75, 75)
        
        out = F.relu(self.conv3_1(out)) # (N, 256, 75, 75)
        out = F.relu(self.conv3_2(out)) # (N, 256, 75, 75)
        out = F.relu(self.conv3_3(out)) # (N, 256, 75, 75)
        out = self.pool3(out) # (N, 256, 38, 38)
        
        out = F.relu(self.conv4_1(out)) # (N, 512, 38, 38)
        out = F.relu(self.conv4_2(out)) # (N, 512, 38, 38)
        out = F.relu(self.conv4_3(out)) # (N, 512, 38, 38)
        conv4_3_feats = out
        out = self.pool4(out) # (N, 512, 19, 19)
        
        out = F.relu(self.conv5_1(out)) # (N, 512, 19, 19)
        out = F.relu(self.conv5_2(out)) # (N, 512, 19, 19)
        out = F.relu(self.conv5_3(out)) # (N, 512, 19, 19)
        out = self.pool5(out) #  pool5 does not reduce dimensions
        
        out = F.relu(self.conv6(out)) # (N, 1024, 19, 19)
        conv7_feats = F.relu(self.conv7(out)) # (N, 1024, 19, 19)
        
        # returning lower level feats
        
        return  conv4_3_feats,conv7_feats
    
    
        
        
        
        