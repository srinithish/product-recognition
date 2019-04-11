# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:45:10 2019

@author: nithish k
"""



import imageManipulations
import plotGridAndBound
import XMLParser
import assigngrid



    
inpFilePic = "C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Train images/images/Arla-Ecological-Medium-Fat-Milk_001.jpg"
inpFileXML = "C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Train images/annotations/xmls/Arla-Ecological-Medium-Fat-Milk_001.xml"
outputImg = 'C:/Users/ntihish/Documents/IUB/Deep Learning/Project/Git Repo/product-recognition/sample_files/resized.jpg'

origImgDict, origObjList = XMLParser.parseXMLtoDict(inpFileXML)

assigngrid


##generate new image
#imageResize(inpFilePic,outputImg,29,29)


gridImg = plotGridAndBound.plotGridOnImg(inpFilePic,20,20,origObjList)
gridImg.savefig(outputImg)
