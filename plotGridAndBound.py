# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:37:12 2019

@author: nithish k
"""



import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def addBoundingBox(plt,objectList,dispClassLabel =True):
    
    """
    args: plt object , objectList
    return: adds bounding boxes
    
    """
    
    ax = plt.gca()
    
    for eachObj in objectList:
        xLeft,yLeft = eachObj['xmin'],eachObj['ymin']
        objWidth = eachObj['xmax'] - eachObj['xmin']
        objHeight  = eachObj['ymax'] - eachObj['ymin']
        centerX = int((eachObj['xmax'] + eachObj['xmin'])/2)
        centerY = int((eachObj['ymax'] + eachObj['ymin'])/2)
        rect = patches.Rectangle((xLeft,yLeft),objWidth,objHeight,
                                 linewidth=5,edgecolor='r',facecolor='none')
        
        plt.scatter(x=[centerX], y=[centerY], c='r', s=40)
        if dispClassLabel == True:
            plt.text(centerX, centerY, eachObj['name'], fontsize = 30 ,color = 'red')
        
        ax.add_patch(rect)
    
    return plt
    
    
    
def plotGridOnImg(filepath,numXGrids,numYGrids,objectList,grid = True,dispClassLabel=  True):
    
    """
    
    """
    
    
    plt.clf()
    img = plt.imread(filepath)
#    img  = np.array(img)
    yPixels,xPixels,channels = img.shape
    
    xSteps,ySteps = xPixels//numXGrids , yPixels//numYGrids
    
#    print(xSteps,ySteps)
    if grid == True:
    ## horizontal lines
        for xCordLine in range(0,xPixels,xSteps):   
            
            plt.axvline(x=xCordLine,linewidth  = 2)
        
        ### vertical lines
        for yCordLine in range(0,yPixels,ySteps):
            plt.axhline(y=yCordLine,linewidth  = 2)
    
    
    addBoundingBox(plt,objectList,dispClassLabel)
#    

    ax = plt.gca()
#    ax.figure.figimage(img,0,0)
#    ax.add_image(img)
    plt.imshow(img)

    
    return plt

if __name__ == '__main__':
    filepath = "C:/Users/ntihish/Pictures/lions-cubs-kenya_53922_990x742.jpg"
    imageDict, objectList = parseXMLtoDict("C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Train images/annotations/xmls/Arla-Ecological-Medium-Fat-Milk_001.xml")
    gridImg = plotGridOnImg(filepath,6,6,objectList)
    gridImg.savefig("griddedImage")
