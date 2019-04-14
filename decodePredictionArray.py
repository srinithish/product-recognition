# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:18:35 2019

@author: nithish k
"""

import imageManipulations
import plotGridAndBound
import XMLParser
import assigngrid
import normalization
import denormalization
import numpy as np
import generateTargetVariable


classMappingDict = {'dog': 0, 'cat' : 1}

def denormPredForEachGrid(image_dict,arrayOf4Elems,grid_row_no,
                          grid_col_no,total_grid_rows,total_grid_cols):
    
    normalized_coords = {}
    normalized_dimensions = {}
    
    normalized_coords['x'] = arrayOf4Elems[0] ##x
    normalized_coords['y'] = arrayOf4Elems[1] ## y
    normalized_dimensions['width'] = arrayOf4Elems[2] ##width
    normalized_dimensions['height'] = arrayOf4Elems[3] ##height
    
    denormalisedBBoxDict = denormalization.getDenormalizedBoxParams(image_dict,
                                            normalized_dimensions,normalized_coords,
                                            grid_row_no,grid_col_no,total_grid_rows,total_grid_cols)
    
    return denormalisedBBoxDict
    
    
    
def decodePredArr(image_dict,predictedArray,classMappingDict):
    
    
    
    """
    """
    
    reverseMappingDict = {value: key for key, value in classMappingDict.items()}
    total_grid_rows,total_grid_cols,total_depth = predictedArray.shape
    
    
    classProbsArray = predictedArray[:,:,5:]
    probOfObjectPresent = predictedArray[:,:,:1] ##preserve 3 dimennsions
    
    probObjTimesClassProb = probOfObjectPresent*classProbsArray
    
    classLabelPredsEachGrid  = np.argmax(probObjTimesClassProb,axis = 2)
    
    
    classProbsEachGrid = np.max(probObjTimesClassProb,axis = 2)
    
    bBoxDims = predictedArray[:,:,1:5]
    
    
    
    
    ### generate object list looping every grid cell
    objectList = []
    for gridRow in range(total_grid_rows):
        for gridCol in range(total_grid_cols):
           arrayOf4Elems =  bBoxDims[gridRow,gridCol,]
           eachObj = denormPredForEachGrid(image_dict,arrayOf4Elems,gridRow,gridCol,
                                            total_grid_rows,total_grid_cols)
            
           
           eachObj['name'] =  reverseMappingDict[classLabelPredsEachGrid[gridRow,gridCol]]
           eachObj['intClass'] = classLabelPredsEachGrid[gridRow,gridCol]
           eachObj['probClass'] = classProbsEachGrid[gridRow,gridCol]
           
           objectList.append(eachObj)
            
    
    return objectList



if __name__ == '__main__':
    
    xmlFile =  "C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Git Repo/product-recognition/sample_files/twoObjectsCorrect.xml"
    
    imgFile = "C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Git Repo/product-recognition/sample_files/twoObjectsCorrect.jpg"
    
    
    imageDict, objectList = XMLParser.parseXMLtoDict(xmlFile)   
    arr = genTargetArray(imgFile,imageDict,objectList,3,3, {'dog': 0, 'cat': 1})
    
    
    objectList = decodePredArr(imageDict,arr,classMappingDict)
    gridImg = plotGridAndBound.plotGridOnImg(imgFile,3,3,objectList)
    gridImg.savefig("griddedImage")
                
    
    
    

