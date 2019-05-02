import numpy as np
from sklearn.metrics import f1_score
import decodePredictionArray
import pickle
import nonMaxSupression
def calculate_accuracy(ground_truth_object_list, pred_object_list):
    """For each image we try to find the ground truch and predicted counts for the classes
    
    Parameters
    ----------
    ground_truth_object_list : [{   'xmin': 330.0,
                                    'xmax': 330.0,
                                    'ymin': 0.0,
                                    'ymax': 0.0,
                                    'name': 'dog',
                                    'intClass': 0,
                                    'probClass': 0.0}]

    pred_object_list :[{            'xmin': 330.0,
                                    'xmax': 330.0,
                                    'ymin': 0.0,
                                    'ymax': 0.0,
                                    'name': 'dog',
                                    'intClass': 0,
                                    'probClass': 0.0}]
    Returns
    -------
    ground_truth_list : List containing the count for each class, index of list means the class
    pred_list : List containing the count for each class, index of list means the class
    """
    length = len(ground_truth_object_list)
    ground_truth_list = np.zeros(length)
    pred_list = np.zeros(length)
    for index in range(length):
        # getting ground truth and predicted value from the list
        ground_truhth = int(ground_truth_object_list[index]['intClass'])
        predicted = int(ground_truth_object_list[index]['intClass'])

        ground_truth_list[ground_truhth] = ground_truth_list[ground_truhth] + 1
        pred_list[predicted] = pred_list[predicted] + 1

    return ground_truth_list, pred_list

def get_r2_score(ground_truth, predicted):
    """Get the weighted f1 score for the model.
    
    Parameters
    ----------
    ground_truth : ndArray of shape (no. of samples x no. of classes)
    predicted : ndArray of shape (no. of samples x no. of classes)

    Returns
    -------
    The R2 score of the model.
    """
    residual = np.sum(np.square(np.subtract(ground_truth, predicted)))
    total = np.sum(np.square(np.subtract(ground_truth, np.mean(ground_truth))))
    return np.subtract(1.0, np.divide(residual, (total + 0.00000000001)))

if __name__ == "__main__":
    
    classMappingDict = {'milk':0,'tomato': 1, 'apple':2 , 'eggs':3 ,'onion': 4,
                    'salt':5, 'yogurt': 6, 'sugar': 7, 'butter': 8, 'orange':9}
    
    ImgDictsPath_True_Path = "renamed_images/resized/annotations/ImageDictsAllFiles.pkl"
    ObjLists_True_Path = "renamed_images/resized/annotations/ObjectListsAllFiles.pkl"
    
    imagefileNames = "renamed_images/resized/annotations/AllFileNames.pkl"
    image_names_list = pickle.load(open(imagefileNames, 'rb'))
    
    predictionArrayPath = "renamed_images/resized/annotations/PredictionArray_correct.pkl"
    
    ##not actually required
    img_dir = "renamed_images/resized/images/"
    image_names_list = [img_dir+img_name for img_name in image_names_list]
    
    
    
    ###load all predictions
    ListOf_imgDicts_true = pickle.load(open(ImgDictsPath_True_Path, 'rb'))
  

    ListOf_ObjLists_GroundTruth = pickle.load(open(ObjLists_True_Path, 'rb'))

    ListOf_PredictionY = pickle.load(open(predictionArrayPath, 'rb'))

    ##loop
    
    ListOf_ObjLists_Pred = []
    
    ovrlpThrs = 0.1
    probThrs = 0.8
    
    for imgDict,predictionArray in zip(ListOf_imgDicts_true,ListOf_PredictionY):
        predObjectList = decodePredictionArray.decodePredArr(imgDict,predictionArray,classMappingDict)
        reducedObjList = nonMaxSupression.non_max_supression_wrapper(predObjectList,classMappingDict,
                                   overlapThresh=ovrlpThrs,probThres=probThrs, checkLabels=True)
        
        ListOf_ObjLists_Pred.append(reducedObjList)

    print(get_r2_score(np.random.rand(100,10), np.random.rand(100,10)))
