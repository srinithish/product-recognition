# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:16:41 2019

@author: gurjaspal
"""
from data_aug.data_aug import * 
import cv2
import matplotlib.pyplot as plt 

def from_object_dict_list(object_dict_list):
    bounding_box_list= []
    for object_dict in object_dict_list:
        bounding_box_list.append([object_dict['xmin'], object_dict['ymin'], object_dict['xmax'], object_dict['ymax']])
    return np.array(bounding_box_list) 

def from_bounding_boxes(bounding_boxes):
    object_dict_list = []
    for bounding_box in bounding_boxes:
        object_dict = {}
        object_dict['xmin'] = bounding_box[0]
        object_dict['xmax'] = bounding_box[1]
        object_dict['ymax'] = bounding_box[2]
        object_dict['ymin'] = bounding_box[3]
        object_dict_list.append(object_dict)
    return object_dict_list


def get_image_after_roation(img, object_dict_list):
    bounding_box_list = from_object_dict_list(object_dict_list)
    img_, bboxes_ = RandomHorizontalFlip(1)(img.copy(), bounding_box_list.copy())
    return img_, from_bounding_boxes(bboxes_)

#TESTING    
img = cv2.imread("experiment_images/two_boxes.jpg")[:,:,::-1]
object_dict1 = {}
object_dict1['xmin'] = 1517
object_dict1['xmax'] = 1645
object_dict1['ymax'] = 1842
object_dict1['ymin'] = 2110

object_dict2 = {}
object_dict2['xmin'] = 2012
object_dict2['xmax'] = 2420
object_dict2['ymax'] = 2302
object_dict2['ymin'] = 2850
object_dict_list =  [object_dict1, object_dict2]
img, new_object_dicts = get_image_after_roation(img, object_dict_list)
print(new_object_dicts)
plt.imshow(img)

