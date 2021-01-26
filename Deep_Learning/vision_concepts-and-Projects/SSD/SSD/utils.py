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

## using List of simple colors:  https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i,k in enumerate(label_map.keys())}


def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    
    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter('object'):
       
        difficult = int(object.find('difficult').text == '1')
        
        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue
        
        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text)-1
        ymin = int(bbox.find('ymin').text)-1
        xmax = int(bbox.find('xmax').text)-1
        ymax = int(bbox.find('ymax').text)-1
        
        boxes.append([xmin,ymin,xmax,ymax])
        labels.append(label_map[label])
        
        difficulties.append(difficult)
        
    return {'boxes': boxes, 'labels':labels, 'difficulties':difficulties}


def create_data_lists(voc07_path, voc07_test, output_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.
    
    :param voc07_path: path to the 'VOC2007' folder
    :param output_folder: folder where the JSONs must be saved
    """
    voc07_path = os.path.abspath(voc07_path)
    voc07_test = os.path.abspath(voc07_test)
    
    train_images = list()
    train_objects = list()
    n_objects = 0
    
    # Training Data
    
    for path in [voc07_path]:
        
        # Find IDs of images in training data
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()
        
        for id in ids:
            # parse annotation's xml file
            objects = parse_annotation(os.path.join(path, 'Annotations', id+'.xml'))
            if len(objects['boxes']) == 0:
                continue
            n_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'JPEGImages', id+'.jpg'))
    
    assert len(train_objects) == len(train_images)
   
    #save file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  
    # saving label map too


    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))
        
    # Test Data
    
    test_images = list()
    test_objects = list()
    n_objects = 0
    
    with open(os.path.join(voc07_test, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()
    
    for id in ids:
        # parse annotation xml file

        objects = parse_annotation(os.path.join(voc07_test, 'Annotations', id+'.xml'))
        if len(objects)==0:
            continue
        test_objects.append(objects)
        n_objects += len(objects)
        test_images.append(os.path.join(voc07_test,'JPEGImages', id+'.jpg'))
        
    assert len(test_objects) == len(test_images)
    
    #saving file
    
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images,j)
    with open(os.path.join(output_folder,'TEST_objects.json'), 'w') as j:
        json.dump(test_objects,j)
        
    print('\nThere are %d test images containing a total of %d objects. Files have been saved to %s.' % (
    len(test_images), n_objects, os.path.abspath(output_folder)))
        
        
        
# create_data_lists('D:\GitHub\Introductory_excersices_ml\Deep_Learning\vision_concepts-and-Projects\SSD\SSD\VOCtrainval', 'new')
        
#create_data_lists(r"D:\GitHub\Introductory_excersices_ml\Deep_Learning\vision_concepts-and-Projects\SSD\SSD\VOCtrainval\VOCdevkit\VOC2007",r"D:\GitHub\Introductory_excersices_ml\Deep_Learning\vision_concepts-and-Projects\SSD\SSD\VOCtest\VOCdevkit\VOC2007",
#                  r"D:\GitHub\Introductory_excersices_ml\Deep_Learning\vision_concepts-and-Projects\SSD\SSD\new")
        
def xy_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).
    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    
    return torch.cat([(xy[:,2:]+xy[:,:2])/2, #c_x,c_y
                      xy[:,2:]-xy[:,:2]],1) # w,h

def cxcy_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).
    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:,:2]-cxcy[:,2:]/2, #xmin,ymin
                      cxcy[:,:2]+cxcy[:,2:]/2],1) #xmax,ymax

def cxcy_gcxgcy(cxcy,priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).
    
    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.
    
    In the model, we are predicting bounding box coordinates in this encoded form.
    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """
    
    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    
    return torch.cat([(cxcy[:,:2]-priors_cxcy[:,:2])/(priors_cxcy[:,2:]/10), # g_c_x,g_c_y 
                      torch.log(cxcy[:,2:]/priors_cxcy[:,2:]*5)],1) # g_w,G-h

def gcxgcy_cxcy(gcxgcy,priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.
    
    They are decoded into center-size coordinates.
    This is the inverse of the function above.
    
    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """
    
    return torch.cat([gcxgcy[:,:2]*priors_cxcy[:,:2]/10 + priors_cxcy[:,:2],  #cxcy
                      torch.exp(gcxgcy[:,2:]/5)*priors_cxcy[:,2:]],1) #wh


def find_interection(set_1,set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
    
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    # PyTorch auto-broadcasts singleton dimensions
    
    lower_bounds = torch.max(set_1[:,:2].unsqueeze(1),set_2[:,:2].unsqueeze(0)) #n1,n2,2
    upper_bounds = torch.min(set_1[:,:2].unsqueeze(1),set_2[:,:2].unsqueeze(0)) #n1,n2,2
    
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)
        
        
        
        
        
        
        
        
        
        
        
        
        
        