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


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    
    # Finding Intersection
    
    intersection = find_interection(set_1,set_2)
    
    # Finding area of each both in both sets
    
    area_set_1 = (set_1[:,2]-set_1[:,0])*(set_1[:,3]-set_1[:,1])
    area_set_2 = (set_2[:,2]-set_2[:,0])*(set_2[:,3]-set_2[:,1])
        
    union = area_set_1.unsqueeze(1) + area_set_2.unsqueeze(0) - intersection
    
    return intersection/union 


# Some augmentation functions below have been adapted from
# From https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    
def expand(image,boxes,filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.
    
    Helps to learn to detect smaller objects.
    
    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    
    # Original dimensions
    original_h = image.size(1)
    original_w = image.size(2)
    
    max_scale = 4
    scale = random.uniform(1,max_scale)
    new_h = int(original_h*scale)
    new_w = int(original_w*scale)
    
    # Creating such imae with filler
    filler = torch.FloatTensor(filler)
    new_image = torch.ones((3,new_h,new_w),dtype=torch.float)*filler.unsqueeze(1).unsqueeze(1)
    
    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0,new_w-original_w)
    right = left+original_w
    top = random.randint(0,new_h-original_h)
    bottom = top+original_h
    new_image[:,top:bottom,left:right] = image

    # Adjusting bounding boxes-just shifting
    new_boxes = boxes + torch.FloatTensor([left,right,top,bottom]).unsqueeze(0)
    
    return new_image,new_boxes
    
def random_crop(image,boxes,labels,difficulties):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.
    
    Note that some objects may be cut out entirely.
    
    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    
    original_h = image.size(1)
    original_w = image.size(2)
    
    # choosing minimum overlap until successful crop is made
    while True:
        # Randomly draw value for minimum overlap
        
        min_overlap = random.choice([.1,.3,.5,.7,.9,.0,None]) # None implies no crop
        
        # if no cropping
        if min_overlap is None:
            return image,boxes,labels,difficulties
        
        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            min_scale = 0.3
            
            scale_h = random.uniform(min_scale,1)
            scale_w = random.uniform(min_scale,1)
            new_h = int(original_h*scale_h)
            new_w = int(original_w*scale_w)
            
            # Aspect ratio be in [0.5,1]
            aspect_ratio = new_h/new_w
            if not 0.5 < aspect_ratio < 2:
                continue
            
            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])
                              
            # calculating jaccard between crop and bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes)
            overlap = overlap.unsqueeze(0)
            
            if overlap.max().item() < min_overlap:
                continue
            
            # Cropping
            new_image = image[:, top:bottom, left:right]
            
            # center of bboxs
            bb_centers = (boxes[:,:2,]+boxes[:,2:])/2.
            
            # bboxes centers in crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom) 
            
            # if no bboxes in croped image 
            if not centers_in_crop.any():
                continue
            
            new_boxes = boxes[centers_in_crop,:]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]
            
            # bboxes coordinates in new crop
            new_boxes[:,:2] = torch.max(new_boxes[:,:2],crop[:2])
            new_boxes[:,:2] -= crop[:2]
            new_boxes[:,2:] = torch.min(new_boxes[:,2:],crop[2:])
            new_boxes[:,2:] -= crop[2:]
            
            return new_image,new_boxes,new_labels,new_difficulties

                
def flip(image,boxes):
    """
    Flip image horizontally.
    
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    
    #flip image
    new_image = FT.hflip(image)
    
    #flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes
    
def resize(image,boxes,dims=(300,300),return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).
    
    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.
    
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    
    #Resize image
    new_image = FT.resize(image,dims)
    
    # resizing bboxes
    
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes

def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.
    
    :param image: image, a PIL Image
    :return: distorted image
    """
    
    new_image = image
    distrotions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]
    random.shuffle(distrotions)
    
    for d in distrotions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image


def transform(image,boxes,labels,difficulties,split):
    """
    Apply the transformations above.
    
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """
    
    assert split in {'TRAIN', 'TEST'}
    
    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # Skip the following operations for evaluation/testing
    if split == 'TRAIN':
        # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
        new_image = photometric_distort(new_image)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
        # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)

        # Randomly crop image (zoom in)
        new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
                                                                         new_difficulties)

        # Convert Torch tensor to PIL image
        new_image = FT.to_pil_image(new_image)
        # Flip image with a 50% chance
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties


def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.
    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.
    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.
    
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))
            
    
def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
    
    
def save_checkpoint(epoch, model, optimizer):
    """
    Save model checkpoint.
    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_ssd300.pth.tar'
    torch.save(state, filename)
    

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        