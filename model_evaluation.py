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
    print(get_r2_score(np.random.rand(100,10), np.random.rand(100,10)))
