# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 01:16:35 2019

@author: nithish k
"""

 
from PIL import Image
import copy
import numpy as np
import math



def imageResize( readFilePath, writeFilePath,resizeWidth,resizeHeight):
    
    
    """
    saves images with required shape and file name
    """
    img = Image.open(readFilePath)
    img = img.resize((resizeWidth,resizeHeight), Image.ANTIALIAS)
    img.save(writeFilePath)
    return True

def rotateImage(readFilePath, writeFilePath,rotation = 0):
    """
    creates a new rotated image
    """
    img = Image.open(readFilePath)
    img = img.rotate(rotation)
    img.save(writeFilePath)
    return True


def _rescaleObjDict(OrigImageDict,OrijObjectList,resizeWidth,resizeHeight):
    
    """
    rescales objDict with resize width and height
    
    returns : alteredObjectList
    """
    
    alteredObjectList = copy.deepcopy(OrijObjectList)
    
    ratioX  = resizeWidth/OrigImageDict['width']
    ratioY =  resizeHeight/OrigImageDict['height']
    
    for eachObj in alteredObjectList:

       eachObj['xmin'] = int(eachObj['xmin'] * ratioX)
       eachObj['ymin'] = int(eachObj['ymin'] * ratioY)
       eachObj['xmax'] = int(eachObj['xmax'] * ratioX)
       eachObj['ymax'] = int(eachObj['ymax'] * ratioY)
       
    
    return alteredObjectList


def rotateObjDict(OrigImageDict,OrijObjectList,rotate = 0):
    
    
    
    pass

def ResizeDict(OrigImageDict,OrijObjectList,dictOfTranslations):
    
    
    """
    dictOfTranslations :  keys: resize,resizeHeight,resizeWidth,rotate,rotateDegree
                        
    """
    
    resizeHeight = dictOfTranslations['resizeHeight']
    resizeWidth = dictOfTranslations['resizeWidth']
    
    ## set orginal false
    ###prepare image dict
    ## change filename if required
    
    AlteredImageDict = copy.deepcopy(OrigImageDict)
    AlteredImageDict['isOriginal'] = False
    
    ### object list
    
    alteredObjectList = _rescaleObjDict(OrigImageDict,OrijObjectList,resizeWidth,resizeHeight)
    AlteredImageDict['width']= resizeWidth
    AlteredImageDict['height'] = resizeHeight
    
    return AlteredImageDict,alteredObjectList 







    
if __name__ == "__main__":
    
    
    ### pipeline
    
    inpFilePic = "./sample_files/twoObjectCorrect.jpg"
    inpFileXML = "./sample_files/twoObjectsCorrect.xml"
    outputImg = 'resized.jpg'
    origImgDict, origObjList = parseXMLtoDict(inpFileXML)
    
    ##generate new image
    imageResize(inpFilePic,outputImg,200,200)
    
    dictOfTranslations = {'resizeWidth' : 200, 'resizeHeight' : 200}
    newImageDict,newObjectList  = ResizeDict(origImgDict,origObjList,dictOfTranslations)
    
    gridImg = plotGridOnImg(outputImg,20,20,newObjectList)
    gridImg.savefig("griddedImage")
    rotateImage(inpFilePic,'rotaion.jpg',360)
    

    