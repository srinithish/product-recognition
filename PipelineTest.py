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
import generateTargetVariable


xNumGrid = 19
yNumGrid = 19
classMappingDict = {'dog': 0, 'cat' : 1}



inpFilePic = "D:/Assignments/Sem 2/Deep learning/Project/Yolo/dl_project/sample_files/twoObjectsCorrect.jpg"
inpFileXML = "D:/Assignments/Sem 2/Deep learning/Project/Yolo/dl_project/sample_files/twoObjectsCorrect.xml"
outputImg = "normalized_img.jpg"

imageDict, ObjList = XMLParser.parseXMLtoDict(inpFileXML)
targetArray = generateTargetVariable.genTargetArray(inpFilePic,imageDict, ObjList,xNumGrid,yNumGrid,classMappingDict)



##generate new image
#imageResize(inpFilePic,outputImg,29,29)

# BB



filepath = inpFilePic
imageDict, objectList = XMLParser.parseXMLtoDict(inpFileXML)
gridImg = plotGridAndBound.plotGridOnImg(filepath,3,3,objectList)
gridImg.savefig("griddedImage.jpg")

gridImg = plotGridAndBound.plotGridOnImg(inpFilePic,20,20,origObjList)
gridImg.savefig(outputImg)


