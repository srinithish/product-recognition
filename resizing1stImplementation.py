# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:31:58 2019

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

inpfilePattern = "C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Train images/images/Arla*"
outputdir = "C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Train images/Reshaped/images/"
outputAnnotations = "C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Train images/Reshaped/annotations"
xmlFiles = "C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Train images/annotations/xmls/Arla*"

AllImageList,AllObjectsList = [],[]
AllImageArray = []
AllImgTargetArray = []

for inpfile,inpXmlFile in zip(sorted(glob.glob(inpfilePattern)),sorted(glob.glob(xmlFiles))):
    
    
    
    outputfileName = os.path.basename(inpfile)
    imgOutputFilePath = outputdir + outputfileName
    imageManipulations.imageResize(inpfile,imgOutputFilePath,342,342)
    
    ### create image X variable
    plt.clf()
    img = plt.imread(imgOutputFilePath)
    AllImageArray.append(img)
    
    
    ##change attributes to reshaped
    origImgDict, origObjList = XMLParser.parseXMLtoDict(inpXmlFile)
    
    dictOfTranslations = {'resizeWidth' : 342, 'resizeHeight' : 342}
    newImageDict,newObjectList  = imageManipulations.ResizeDict(origImgDict,origObjList,dictOfTranslations)
    
    AllImageList.append(newImageDict)
    AllObjectsList.append(newObjectList)
    
    ### create img targets
    eachImgTarget = generateTargetVariable.genTargetArray('',newImageDict,newObjectList,19,19, {'Milk': 0})
    AllImgTargetArray.append(eachImgTarget)


import pickle
with open(outputAnnotations+'/resizedImageDictsAllFiles.pkl', 'wb') as f:
   pickle.dump(AllImageList, f)
with open(outputAnnotations+'/resizedObjectListsAllFiles.pkl', 'wb') as f:
   pickle.dump(AllObjectsList, f)

with open(outputAnnotations+'/resizedAllImgArray.pkl', 'wb') as f:
   pickle.dump(AllImageArray, f)
   
with open(outputAnnotations+'/resizedAllTargetArray.pkl', 'wb') as f:
   pickle.dump(AllImgTargetArray, f)
   
   
   
   
   

   
###image dicts   
classMappingDict = {'Milk':0}
ImgDictsPath = "C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Train images/Reshaped/annotations/resizedImageDictsAllFiles.pkl"
with open(ImgDictsPath, 'rb') as f:
  imageDicts = pickle.load(f)
    
imgFile = "C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Train images/Reshaped/images/Arla-Ecological-Medium-Fat-Milk_002.jpg"

predictionPath = "C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Train images/Reshaped/annotations/PredictionArray.pkl"

with open(predictionPath, 'rb') as f:
  PredictionY = pickle.load(f) 

objectList = decodePredictionArray.decodePredArr(imageDicts[1],PredictionY[1],classMappingDict)

boxes, probs, labels = nonMaxSupression.input_to_nms(objectList)
boxes,labels = nonMaxSupression.non_max_suppression(boxes, probs, labels, overlapThresh=0.5, probThres=0.22,checkLabels=True)
newObjList  = nonMaxSupression.convert_op_nms(boxes,labels,classMappingDict)

gridImg = plotGridAndBound.plotGridOnImg(imgFile,19,19,newObjList,grid =False)
gridImg.savefig("griddedImage")
                  

