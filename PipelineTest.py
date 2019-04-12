# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:45:10 2019

@author: nithish k
"""



import imageManipulations
import plotGridAndBound
import XMLParser
import assigngrid
import normalization
import denormalization



    
inpFilePic = "D:/Assignments/Sem 2/Deep learning/Project/Yolo/dl_project/sample_files/twoObjectsCorrect.jpg"
inpFileXML = "D:/Assignments/Sem 2/Deep learning/Project/Yolo/dl_project/sample_files/twoObjectsCorrect.xml"
outputImg = "normalized_img.jpg"

origImgDict, origObjList = XMLParser.parseXMLtoDict(inpFileXML)
print(origImgDict)

##generate new image
#imageResize(inpFilePic,outputImg,29,29)

# BB
box_params = normalization.getNormalizedBoxParams(origImgDict,origObjList[1],1,2,3,3)


filepath = inpFilePic
imageDict, objectList = XMLParser.parseXMLtoDict(inpFileXML)
gridImg = plotGridAndBound.plotGridOnImg(filepath,3,3,objectList)
gridImg.savefig("griddedImage.jpg")

gridImg = plotGridAndBound.plotGridOnImg(inpFilePic,20,20,origObjList)
gridImg.savefig(outputImg)


