# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:33:15 2019

@author: nithish k
"""

import xmltodict

def parseXMLtoDict(fileName):
    
    """
    args:
        filename: takes absolute path of the file
    
    returns:
        imageDict : keys: width,height,depth,path,filename
        objectList :  list of object dicts {'name': 'lion', 'xmin': '220', 'ymin': '181', 'xmax': '853', 'ymax': '785'}
            
    """
    
    with open(fileName,'r') as f:
        dictOfAnotations = xmltodict.parse(f.read())
        f.close()
        
    ##image dict generation
    imageDict = dictOfAnotations['annotation']['size']
    imageDict['path'] = dictOfAnotations['annotation']['path']
    imageDict['filename'] = dictOfAnotations['annotation']['filename']
    
    objectList = []
    ###object dict
    for eachObj in dictTemp['annotation']['object']:
        objectDict = {}
        objectDict['name'] = eachObj['name']
        
        objectDict.update(eachObj['bndbox']) 
        
        objectList.append(objectDict)
    
    
    return imageDict,objectList
     

if __name__ == '__main__':
    
    imageDict, objectList = parseXMLtoDict("C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Git Repo/product-recognition/twoObjects.xml")
