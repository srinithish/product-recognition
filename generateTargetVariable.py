# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:06:37 2019

@author: nithish k
"""

import imageManipulations
import plotGridAndBound
import XMLParser
import assigngrid


xmlFile =  "C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Git Repo/product-recognition/sample_files/twoObjectsCorrect.xml"

imageDict, objectList = XMLParser.parseXMLtoDict(xmlFile)


def genTargetArray(imgFile,imageDict, objectList,xNumGrid,yNumGrid,classMappingDict):
    
    """
    """
    
    numClasses = len(classMappingDict)
    depth = 1+ 4 + numClasses  ## pc + 4 bb dims + numclasses
    
    targetArray = np.zeros((xNumGrid, yNumGrid , depth ))
    for eachObj in objectList:
        gridRow,gridCol = assigngrid.assign_grid(imageDict,eachObj,xNumGrid,yNumGrid)
        objVector = np.zeros((depth))
        objVector[0] = 1 ### assign probOfObj
        ### call normalise dims
        objVector[1:5] = normwlise
        targetArray[gridRow,gridCol,:]
        
    
    
    
    
    
    