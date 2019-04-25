# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:52:16 2019

@author: nithish k
"""



import glob
import os

import imageManipulations
import plotGridAndBound
import XMLParser
import assigngrid
import normalization
import denormalization
import generateTargetVariable
import decodePredictionArray

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
import nonMaxSupression

###image dicts   
"""
Note: need to change class mapping dict as required as global variable 
"""
classMappingDict = {'milk':0,'tomato': 1, 'apple':2 , 'eggs':3 ,'onion': 4,
                    'salt':5, 'yogurt': 6, 'sugar': 7, 'butter': 8, 'orange':9}

###has all of the Image informations as list of dicts



###plot at image level


def visualisePredictions(trueImgDict,trueObjList,imgPath,eachPredictionArray,
                         overlapThresh=0.5,probThres=0):
    """
    Should shange the thresholds if required
    change grid size as well if needed
    """
    
    global classMappingDict
    
    pred_objectList = decodePredictionArray.decodePredArr(trueImgDict,
                                                          eachPredictionArray,
                                                          classMappingDict)
    
    
    postNonMaxPredObjList = nonMaxSupression.non_max_supression_wrapper(pred_objectList,classMappingDict,
                                                                     overlapThresh,probThres)
    
    gridImg = plotGridAndBound.plotGridOnImg(imgPath,19,19,postNonMaxPredObjList,
                                             grid =False, dispClassLabel = False)

    
    plt.figure()
    
    
    


def visualise_preds_for_set_of_images(imgfilePattern,
                                      ImgDictsPath_True_Path,
                                      ObjLists_True_Path,
                                      predictionArrayPath,
                                      overlapThresh=0.5,probThres=0,
                                      maxImagesToPlot = 10):
    
    """
    requires :
    img dict pkl file
    objectList pkl file
    predictions pkl file
    imagepaths list
    """
    global classMappingDict
    imgFilePaths_list = sorted(glob.glob(imgfilePattern))


    
    ListOf_imgDicts_true = pickle.load(open(ImgDictsPath_True_Path, 'rb'))
  

    ListOf_ObjLists_true = pickle.load(open(ObjLists_True_Path, 'rb'))

    ListOf_PredictionY = pickle.load(open(predictionArrayPath, 'rb'))




    for index in range(maxImagesToPlot):
        
        visualisePredictions(ListOf_imgDicts_true[index],
                             ListOf_ObjLists_true[index],imgFilePaths_list[index],
                             ListOf_PredictionY[index],
                             overlapThresh=0.5,probThres=0)
        






#maxE = -float("Inf")
#for index,dic in enumerate(pred_objectList):
# 
#    if dic['ObjectnessProb'] >= maxE:
#        maxE = dic['ObjectnessProb']
#        maxIndex = index        
#print(maxIndex)
#
#pred_objectList[maxIndex]
        
        
if __name__ == '__main__':
    
    
    """
    Pretraining informations
    """
    ##pickle files
    ImgDictsPath_True_Path = "C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Train images/Reshaped/annotations/resizedImageDictsAllFiles.pkl"
    ObjLists_True_Path = "C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Train images/Reshaped/annotations/resizedObjectListsAllFiles.pkl"
    ###gets all the image as list
    imagefilePattern = "C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Train images/images/Arla*"
    
    
    """
    Prediction informations
    """
    #### prediction array pickle path
    predictionArrayPath = "C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Train images/Reshaped/annotations/PredictionArray.pkl"

    visualise_preds_for_set_of_images(imagefilePattern,
                                      ImgDictsPath_True_Path,
                                      ObjLists_True_Path,
                                      predictionArrayPath,
                                      overlapThresh=0.5,probThres=0,
                                      maxImagesToPlot = 3)
    
    
    
    
    