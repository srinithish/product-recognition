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
from image_operations import get_image_after_rotation



inpfilePattern = "./renamed_images/Original Images/images/*"
inpXmlFiles = "./renamed_images/Original Images/xmls/*"

outputdirImages = "./renamed_images/augmented data/images/"
outputAnnotations = "./renamed_images/augmented data/annotations/"




AllImageDictList,AllObjectsList = [],[]
AllImageAsNPArray = []
AllImgTargetArray = []
imgfileNamesList = []

classMappingDict = {'milk':0,'tomato': 1, 'apple':2 , 'eggs':3 ,'onion': 4,
                    'salt':5, 'yogurt': 6, 'sugar': 7, 'butter': 8, 'orange':9}

GRID_SIZE = 19



for inpfile,inpXmlFile in zip(sorted(glob.glob(inpfilePattern)),sorted(glob.glob(inpXmlFiles))):
    
    
    angles = [90, 180, 270]
    origBasefileName = os.path.basename(inpfile)
    origBasefileName_noext,extension = os.path.splitext(origBasefileName)
#    imgOutputFilePath = outputdir + outputfileName
#    imageManipulations.imageResize(inpfile,imgOutputFilePath,342,342)
    print(origBasefileName)
    ### create image X variable
    plt.clf()
 
    orig_img_array = plt.imread(inpfile)
    
    
    
    
    
    """
    Original image all as imgNumpyArray,imgDict,objList,targetArray, imgfileName
    """
#    AllImageAsNPArray.append(orig_img_array)

    origImgDict, origObjList = XMLParser.parseXMLtoDict(inpXmlFile)
    
    AllImageDictList.append(origImgDict)
    AllObjectsList.append(origObjList)
    
    eachImgTarget = generateTargetVariable.genTargetArray('',origImgDict,origObjList,
                                                          GRID_SIZE,GRID_SIZE,classMappingDict )
    AllImgTargetArray.append(eachImgTarget)
    imgfileNamesList.append(origBasefileName)
    
    plt.imsave(outputdirImages+origBasefileName,orig_img_array)
    
    
    
    """
    
    """
    """
    for reszing 
    """
#    dictOfTranslations = {'resizeWidth' : 342, 'resizeHeight' : 342}
#    newImageDict,newObjectList  = imageManipulations.ResizeDict(origImgDict,origObjList,dictOfTranslations)
    
    
#    To rotate images and get there image dict and object list
#    Then appending to the list of image dict and object dict
    
    """
    rotations
    
    """
    
#    
    for angle in angles:
        rotImageAsArray, rotImageDict, rotObjectList = get_image_after_rotation(orig_img_array, origImgDict, 
                                                                         angle, origObjList)
    

        AllImageAsNPArray.append(rotImageAsArray)
        AllImageDictList.append(rotImageDict)
        AllObjectsList.append(rotObjectList)
        
        eachImgTarget = generateTargetVariable.genTargetArray('',rotImageDict,
                                                              rotObjectList,
                                                              GRID_SIZE,GRID_SIZE, 
                                                              classMappingDict)
        AllImgTargetArray.append(eachImgTarget)
        
        newBaseFileName = origBasefileName_noext+'_'+str(angle)+extension ##append degree
        
        
        imgfileNamesList.append(newBaseFileName)
        plt.imsave(outputdirImages+newBaseFileName,rotImageAsArray)
        
        


with open(outputAnnotations+'/ImageDictsAllFiles.pkl', 'wb') as f:
   pickle.dump(AllImageDictList, f)

with open(outputAnnotations+'/ObjectListsAllFiles.pkl', 'wb') as f:
   pickle.dump(AllObjectsList, f)

#with open(outputAnnotations+'/AllImgNPArray.pkl', 'wb') as f:
#   pickle.dump(AllImageAsNPArray, f)
   
with open(outputAnnotations+'/AllTargetArray.pkl', 'wb') as f:
   pickle.dump(AllImgTargetArray, f)

with open(outputAnnotations+'/AllFileNames.pkl', 'wb') as f:
   pickle.dump(imgfileNamesList, f)
#   

   

   
   


   


                  
