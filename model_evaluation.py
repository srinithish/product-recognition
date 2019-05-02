import numpy as np
from sklearn.metrics import f1_score

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

def get_weighted_f1_score(ground_truth, predicted):
    """Get the weighted f1 score for the model.
    
    Parameters
    ----------
    ground_truth : ndArray of shape (no. of samples x no. of classes)
    predicted : ndArray of shape (no. of samples x no. of classes)

    Returns
    -------
    The F1 score of the model.
    """
    return f1_score(np.sum(ground_truth, axis = 0).reshape(10,1), np.sum(predicted, axis = 0).reshape(10,1), average='weighted')

