# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 18:23:37 2019

@author: Amit

Source: https://github.com/bruceyang2012/nms_python/blob/master/nms.py
"""

import numpy as np


def input_to_nms(object_dict):
    '''
    For eg:
    object_dict =  
    {'center_x': 330,
      'center_y': 0,
      'box_h': 0,
      'box_w': 0,
      'xmin': 330.0,
      'xmax': 330.0,
      'ymin': 0.0,
      'ymax': 0.0,
      'name': 'dog',
      'intClass': 0,
      'probClass': 0.0}
    
    returns boxes,probs,labels
    boxes.shape => # of grids x 4 box params (xmin,ymin,xmin,xmax)
    probs.shape => # of grids  (This probability is Po*(prob of that class))
    labels.shape => # of grids  (Gives the class label predicted for that gridcell)
    '''
    boxes = [[i["xmin"],i["ymin"],i["xmax"],i["ymax"]] for i in object_dict]
    boxes = np.array(boxes)
    
    probs = [i["probClass"] for i in object_dict]
    probs = np.array(probs)
    
    labels = [i["intClass"] for i in object_dict]
    labels = np.array(labels)
    
    return boxes,probs,labels
    


def convert_op_nms(boxes,labels,classMappingDict): 
    '''
    classMappingDict = {'dog': 0, 'cat' : 1}
    '''
    assert boxes.shape[0] == labels.shape[0]
    n = boxes.shape[0]
    output = []
    
    reverseMappingDict = {value: key for key, value in classMappingDict.items()}
    
    for i in range(n):
        box_dict = {}
        box_dict["xmin"] = boxes[i][0]
        box_dict["ymin"] = boxes[i][1]
        box_dict["xmax"] = boxes[i][2]
        box_dict["ymax"] = boxes[i][3]
        box_dict["intClass"] = labels[i]
        box_dict["name"] = reverseMappingDict[labels[i]]
        output.append(box_dict)
    
    return output
    
    
    
def non_max_suppression(boxes, probs, labels, overlapThresh=0.5, probThres=0.1,checkLabels=True):
    """
    Assume the inputs are np arrays
    boxes.shape => # of grids x 4 box params (xmin,ymin,xmin,xmax) 
    probs.shape => # of grids  (This probability is Po*(prob of that class))
    labels.shape => # of grids  (Gives the class label predicted for that gridcell)
    
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    boxes = boxes.astype("float")
    
    # Delete boxes that have very low probability
    keep_indices = np.where(probs >= probThres)
    boxes = boxes[keep_indices]
    probs = probs[keep_indices]
    labels = labels[keep_indices]
    
    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1) * (y2 - y1)

    # To handle area == 0, add a small value
    epsilon = 10**-50
    # sort the indexes, this gives the grid cell with highest probability of having an object
    idxs = np.argsort(probs) # [1,2,3..9]

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        # xx1 and yy1 are the overlapping box's left corner
        # We compare element-wise the max of the all the boxes wrt last
        xx1 = np.maximum(x1[i], x1[idxs[:last]]) # xx1.shape => no. of boxes in consideration - 1
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        # If the boxes have absolutely no overlap the width and height of the 
        # overlapping box will be zero
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # compute the ratio of overlap
        # Overap = overlap box area / area of box under consideration
#        print(w.shape,h.shape)
        overlap = (w * h) / (area[idxs[:last]] + epsilon)

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        if checkLabels:
            overlap_idxs = overlap >= overlapThresh
            same_labels = labels[:last] == labels[i]
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(np.logical_and(overlap_idxs,same_labels))[0])))
        else:
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))
            

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int"), labels[pick]

def non_max_supression_wrapper(object_dict,classMappingDict,overlapThresh=0.5,probThres=0.1, checkLabels=True):
    boxes,probs,labels = input_to_nms(object_dict)
    boxes,labels = non_max_suppression(boxes, probs, labels, 
                                       overlapThresh=overlapThresh, probThres=probThres,checkLabels=checkLabels)
    output = convert_op_nms(boxes,labels,classMappingDict)
    return output
    

if __name__  == "__main__":
    import imageManipulations
    import plotGridAndBound
    import XMLParser
    import assigngrid
    import normalization
    import denormalization
    import generateTargetVariable
    
    
    xNumGrid = 19
    yNumGrid = 19
    classMappingDict = {'lion': 0, 'cat' : 1}
    
    inpFilePic = "D:/Assignments/Sem 2/Deep learning/Project/Yolo/dl_project/testing/resized.jpg"
    inpFileXML = "D:/Assignments/Sem 2/Deep learning/Project/Yolo/dl_project/testing/resized.xml"
    
    imageDict, ObjList = XMLParser.parseXMLtoDict(inpFileXML)
    targetArray = generateTargetVariable.genTargetArray(inpFilePic,imageDict, ObjList,xNumGrid,yNumGrid,classMappingDict)
    
    
    
    