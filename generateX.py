# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:32:02 2019

@author: Amit
"""
import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def generateX(filepath,grayscale=False):
    '''
    filepath 
    '''
    image = Image.open(filepath)
    
    if grayscale == True:
        image = image.convert("L")
    arr = np.asarray(image)
    
    return arr


generateX('sample_files/1.jpg',grayscale=True).shape
