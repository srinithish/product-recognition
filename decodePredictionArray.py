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



def decodePredArr(predictedArray):
    
    
    
    """
    """
    classPredictionArray = predictedArray[:,:,5:]
    probOfObjectPresent = predictedArray[:,:,0]
    
    probObjTimesClassProb = probOfObjectPresent*classPredictionArray
    
    classLabelPreds  = np.argmax(probObjTimesClassProb,axis = 2)
    
    denormalization
        
    

    

