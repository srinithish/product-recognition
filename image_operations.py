# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:16:41 2019

@author: gurjaspal
"""
from data_aug.data_aug import RandomHorizontalFlip, RandomRotate , Rotate, Shear, Resize, draw_rect
import cv2
import matplotlib.pyplot as plt 
import numpy as np
import copy

def from_object_dict_list(object_dict_list):
    bounding_box_list= []
    for object_dict in object_dict_list:
        bounding_box_list.append([float(object_dict['xmin']), float(object_dict['ymin']), float(object_dict['xmax']), float(object_dict['ymax']), 3])
    return np.array(bounding_box_list) 

def from_bounding_boxes(bounding_boxes, _object_dict_list):
    object_dict_list = []
    for index, bounding_box in enumerate(bounding_boxes):
        object_dict = {}
        object_dict['xmin'] = bounding_box[0]
        object_dict['ymin'] = bounding_box[1]
        object_dict['xmax'] = bounding_box[2]
        object_dict['ymax'] = bounding_box[3]
        object_dict['name'] = _object_dict_list[index]['name']
        object_dict_list.append(object_dict)
    return object_dict_list


def get_image_after_horizontal_flip(img, object_dict_list):
    bounding_box_list = from_object_dict_list(object_dict_list)
    img_, bboxes_ = RandomHorizontalFlip(1)(img.copy(), bounding_box_list.copy())
    return img_, from_bounding_boxes(bboxes_, object_dict_list )

def get_image_after_rotation(img, image_dict, angle, object_dict_list):
    bounding_box_list = from_object_dict_list(object_dict_list)
    img_, bboxes_ = Rotate(angle)(img.copy(), bounding_box_list.copy())
    image_dict['width'] = img_.shape[0]
    image_dict['height'] = img_.shape[1] 
    return img_, image_dict, from_bounding_boxes(bboxes_, object_dict_list)

def get_image_after_shear(img, shear, object_dict_list):
    bounding_box_list = from_object_dict_list(object_dict_list)
    img_, bboxes_ = Shear(shear)(img.copy(), bounding_box_list.copy())
    return img_, from_bounding_boxes(bboxes_, object_dict_list)

def get_image_after_resize(img, size_of_one_side, object_dict_list):
    bounding_box_list = from_object_dict_list(object_dict_list)
    img_, bboxes_ = Resize(size_of_one_side)(img.copy(), bounding_box_list.copy())
    return img_, from_bounding_boxes(bboxes_, object_dict_list)

if __name__ == '__main__':
#TESTING    
    img = cv2.imread("experiment_images/two_boxes.jpg")[:,:,::-1]
    object_dict1 = {}
    object_dict1['name'] = 'eggs'
    object_dict1['xmin'] = 1517
    object_dict1['ymin'] = 1645
    object_dict1['xmax'] = 1842
    object_dict1['ymax'] = 2110
    
    object_dict2 = {}
    object_dict2['name'] = 'tomato'
    object_dict2['xmin'] = 2012
    object_dict2['ymin'] = 2420
    object_dict2['xmax'] = 2302
    object_dict2['ymax'] = 2850
    
    image_dict = {}
    image_dict['path'] = "path/to/file"
    image_dict['width'] = img.shape[0]
    image_dict['height'] = img.shape[1]
    image_dict['depth'] = 3
    object_dict_list =  [object_dict1, object_dict2]
    
    
    # img_new, new_object_dicts = get_image_after_horizontal_flip(img.copy(),copy.deepcopy(object_dict_list))
    # plt.figure()
    # plotted_img = draw_rect(img_new, from_object_dict_list(new_object_dicts))
    # plt.imsave('after_horizontal_flip',plotted_img)
    
    
    img_new, image_dict, new_object_dicts = get_image_after_rotation(img.copy(), image_dict, 90, copy.deepcopy(object_dict_list))
    plt.figure()
    plotted_img = draw_rect(img_new, from_object_dict_list(new_object_dicts))
    plt.imsave('after_rotation',plotted_img)
    
    # img_new, new_object_dicts = get_image_after_shear(img.copy(), 0.2, copy.deepcopy(object_dict_list))
    # plt.figure()
    # plotted_img = draw_rect(img_new, from_object_dict_list(new_object_dicts))
    # plt.imsave('after_shear',plotted_img)
    
    # img_new, new_object_dicts = get_image_after_resize(img.copy(),300, copy.deepcopy(object_dict_list))
    # plt.figure()
    # plotted_img = draw_rect(img_new, from_object_dict_list(new_object_dicts))
    # plt.imsave('after_resize',plotted_img)
    #plt.imshow(img)
    
