# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:06:37 2019

@author: nithish k
"""


import imageManipulations
import plotGridAndBound
import XMLParser
import assigngrid
import normalization
import denormalization
import numpy as np



def genTargetArray(imgFile,imageDict, objectList,xNumGrid,yNumGrid,classMappingDict):
    
    """
    args: imgFile,imageDict, objectList,xNumGrid,yNumGrid, classMappingDict = {'dog': 0, 'cat' : 1}
    
    returns : target Array for each image
    """
    
    numClasses = len(classMappingDict)
    depth = 1 + 4 + numClasses  ## pc + 4 bb dims + numclasses
    startIndexClass = 5
    
    targetArray = np.zeros((xNumGrid, yNumGrid , depth ))
    
    
    
    ### form object vector
    for eachObj in objectList:
        
        gridRow,gridCol = assigngrid.assign_grid(imageDict,eachObj,xNumGrid,yNumGrid)
        
        objVector = np.zeros((depth))
        
        objVector[0] = 1 ### assign probOfObj
        
        ### call normalise dims
        ##image_dict,object_dict,grid_row_no,grid_col_no,no_of_row_grids,no_of_col_grids
        normalisedBoxParams = normalization.getNormalizedBoxParams(imageDict,
                                                                   eachObj,gridRow,gridCol,
                                                                   xNumGrid,yNumGrid)
        
        ###update box params to vector
        objVector[1] = normalisedBoxParams['bx']
        objVector[2] = normalisedBoxParams['by']
        objVector[3] = normalisedBoxParams['bh']
        objVector[4] = normalisedBoxParams['bw']
        
        
        ###update class vector
        numericClassValue = classMappingDict[eachObj['name']] ## gets the numeric value of the class
        objVector[startIndexClass+numericClassValue] = 1
        
        
        targetArray[gridRow,gridCol] = objVector
#        
        
    return targetArray

if __name__ == '__main__':
    xmlFile =  "C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Git Repo/product-recognition/sample_files/twoObjectsCorrect.xml"

    imageDict, objectList = XMLParser.parseXMLtoDict(xmlFile)

    arr = genTargetArray('',imageDict,objectList,3,3, {'dog': 0, 'cat': 1})
    

    
    
    