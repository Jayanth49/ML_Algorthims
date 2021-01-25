# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:48:32 2021

@author: Jayanth
"""

import os
import json

import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Labels:(tuple of length 20)

voc_labels =  ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

label_map = {k : v+1 for v,k in enumerate(voc_labels)}
label_map['background'] = 0

rev_label_map = {v : k for k,v in label_map.items()} # Inverse Mapping

# using List of simple colors:  https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i,k in enumerate(label_map.keys())}

