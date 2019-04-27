# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:01:55 2019

@author: nithish k
"""

import plotGridAndBound
import pickle
import decodePredictionArray

inpfilePattern = "./Train images/New Cutom Dataset Trial/OrigData/images/*"
inpXmlFiles = "./Train images/New Cutom Dataset Trial/OrigData/xmls/*"
outputdirImages = "./Train images/New Cutom Dataset Trial/augmented data/images/"
outputAnnotations = "./Train images/New Cutom Dataset Trial/augmented data/annotations/"

imgFilepath = "./Train images/New Cutom Dataset Trial/augmented data/images/0_90.jpg"

with open(outputAnnotations+'/ImageDictsAllFiles.pkl', 'rb') as f:
   imgDicts = pickle.load(f)

with open(outputAnnotations+'/ObjectListsAllFiles.pkl', 'rb') as f:
   objLists = pickle.load(f)

with open(outputAnnotations+'/AllTargetArray.pkl', 'rb') as f:
   AllImgTargetArray = pickle.load(f)
   
gridImg = plotGridAndBound.plotGridOnImg(imgFilepath,19,19,objLists[1])
gridImg.savefig("griddedImage")



objectList = decodePredictionArray.decodePredArr(imgDicts[1],AllImgTargetArray[1],classMappingDict)
gridImg = plotGridAndBound.plotGridOnImg(imgFilepath,19,19,objectList,dispClassLabel=False)

gridImg.savefig("griddedImage")


inpfile = "./Train images/New Cutom Dataset Trial/resized/images/0_270.jpg"
inputAnnotations = "./Train images/New Cutom Dataset Trial/resized/annotations/"

with open(inputAnnotations+'/ImageDictsAllFiles.pkl', 'rb') as f:
   imgDicts = pickle.load(f)

with open(inputAnnotations+'/ObjectListsAllFiles.pkl', 'rb') as f:
   objLists = pickle.load(f)

gridImg = plotGridAndBound.plotGridOnImg(inpfile,19,19,objLists[3])
gridImg.savefig("griddedImage")














