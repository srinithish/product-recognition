# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:25:23 2019

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
from image_operations import get_image_after_rotation

classMappingDict = {'milk':0,'tomato': 1, 'apple':2 , 'eggs':3 ,'onion': 4,
                    'salt':5, 'yogurt': 6, 'sugar': 7, 'butter': 8, 'orange':9}


##change for size
dictOfTranslations = {'resizeWidth' : 342, 'resizeHeight' : 342}

rWidth,rHeight = dictOfTranslations['resizeWidth'],dictOfTranslations['resizeHeight']

GRID_SIZE = 19


### change dirs
inpImagesfileDir = "./Train images/New Cutom Dataset Trial/augmented data/images/"
inpAnnotations = "./Train images/New Cutom Dataset Trial/augmented data/annotations/"

outputdirImages = "./Train images/New Cutom Dataset Trial/resized/images/"
outputAnnotations = "./Train images/New Cutom Dataset Trial/resized/annotations/"





with open(inpAnnotations+'/ImageDictsAllFiles.pkl', 'rb') as f:
   AllImageDictList = pickle.load(f)

with open(inpAnnotations+'/ObjectListsAllFiles.pkl', 'rb') as f:
   AllObjectsList = pickle.load(f)

with open(inpAnnotations+'/AllTargetArray.pkl', 'rb') as f:
   AllImgTargetArray = pickle.load(f)

with open(inpAnnotations+'/AllFileNames.pkl', 'rb') as f:
   imgfileNamesList = pickle.load(f)



newAllImageDictList,newAllObjectsList = [],[]
newAllImageAsNPArray = []
newAllImgTargetArray = []
newImgfileNamesList = []


for index,inpBasefileName in enumerate(imgfileNamesList):
    
    inpImageFile= inpImagesfileDir+inpBasefileName
    outputImagepath = outputdirImages+inpBasefileName
    
    
    imageManipulations.imageResize(inpImageFile,outputImagepath,rWidth,rHeight) ##saves new image
    
    new_img_array = plt.imread(outputImagepath)
    
    origImgDict = AllImageDictList[index]
    origObjList = AllObjectsList[index]
    
    
    
    newImageDict,newObjectList  = imageManipulations.ResizeDict(origImgDict,origObjList,dictOfTranslations)
    
    newEachImgTarget = generateTargetVariable.genTargetArray('',newImageDict,newObjectList,
                                                          GRID_SIZE,GRID_SIZE,classMappingDict )
    
    
    newAllImageAsNPArray.append(new_img_array)
    newAllImageDictList.append(newImageDict)
    newAllObjectsList.append(newObjectList)
    
    newAllImgTargetArray.append(newEachImgTarget)
    newImgfileNamesList.append(outputImagepath)


with open(outputAnnotations+'/ImageDictsAllFiles.pkl', 'wb') as f:
   pickle.dump(newAllImageDictList, f)

with open(outputAnnotations+'/ObjectListsAllFiles.pkl', 'wb') as f:
   pickle.dump(newAllObjectsList, f)

with open(outputAnnotations+'/AllImgNPArray.pkl', 'wb') as f:
   pickle.dump(newAllImageAsNPArray, f)
   
with open(outputAnnotations+'/AllTargetArray.pkl', 'wb') as f:
   pickle.dump(newAllImgTargetArray, f)

with open(outputAnnotations+'/AllFileNames.pkl', 'wb') as f:
   pickle.dump(newImgfileNamesList, f)