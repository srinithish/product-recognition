# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 18:32:09 2019

@author: gurjaspal
"""
from sklearn.metrics import mean_squared_error
import numpy as np

def get_details(row, col, tensor):
    class_prob = tensor[row][col][0]
    x = tensor[row][col][1]
    y = tensor[row][col][2]
    w = tensor[row][col][3]
    h = tensor[row][col][4]
    return class_prob, x, y, w, h
    
def yolo_loss(predicted, ground_truth):
    rows = predicted.shape[0]
    cols = predicted.shape[1]
    
    localization_loss = 0
    lambda_cordinates = 0
    dimension_loss = 0
    for row in range(rows):
        for col in range(cols):
            
            predict_class_prob, predicted_x, predicted_y, predicted_w, predicted_h = get_details(row,col, predicted)
            true_class_prob, true_x, true_y, true_w, true_h = get_details(row,col, ground_truth)
            
            localization_loss = localization_loss + mean_squared_error([true_x, true_y], [predicted_x, predicted_y])
            dimension_loss  = dimension_loss + mean_squared_error([np.sqrt(true_w), np.sqrt(true_h)], [np.sqrt(predicted_w), np.sqrt(predicted_h)])
            
            
    