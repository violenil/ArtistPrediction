import random
from typing import List, Dict, Tuple


def artistPredictor(batch: List, list_artist_frequency: List) -> List:
    list_of_predicted_labels = []
    for i in range(len(batch)):
        list_of_predicted_labels.append(random.choice(list_artist_frequency))
    return list_of_predicted_labels


def extract_labels(batch: List) -> List:
    """
    Called from train or test neural network functions and it extracts the labels
    from the respective batch, and appends them to a separate list.
    """
    list_labels = []
    for song in batch:
        output = song[0]
        list_labels.append(output)
    return list_labels


def evaluate_predictions(list_of_labels: List, list_of_predicted_labels: List, list_of_artists:List) -> Dict:
    """
    This function takes three lists as parameters.

    :param list_of_labels: this contains all the correct labels of the dataset. E.g. [2,2,3,4,8,6,1,8]
    :param list_of_predicted_labels: this contains all the labels predicted by our classifier. E.g. [2,5,8,1,8,4,4,6]
    :param list_of_artists: The list of all the artists in our dataset. E.g. [1,2,3,4,5,6,7,8]
    :return: a dictionary of TP, FP, FN and TN which looks like {TP:[], FP:[], FN:[], TN:[]}
    """
    dict_of_results = {'TP': [], 'FP': [], 'FN': [], 'TN': []}
    for artist in list_of_artists:  # looping through each class (artist) at a time
        tp = 0
        fp = 0 
        fn = 0
        tn = 0
        indexesPred = [i for i, x in enumerate(list_of_predicted_labels) if x == artist]
        indexesGold = [i for i, x in enumerate(list_of_labels) if x == artist] # retrieving all indexes of the artist in predicted and gold, to compare them
        for index in indexesPred:   # going through all predicted indexes for this artist
            if index in indexesGold:    # if we find a match, then tp is increased by 1
                tp += 1
            else:
                fp += 1         # if it doesnt occur, we have a false positive
        for index in indexesGold:   # going through all gold indexes to see if we missed any
            if index not in indexesPred:    # if they dont occur in predicted index list, then we have a false negative
                fn += 1
        dict_of_results['TP'].append(tp)
        dict_of_results['FP'].append(fp)
        dict_of_results['FN'].append(fn)
        dict_of_results['TN'].append(tn) # Note: I haven't given any thought to tn, how to calculate this? Does it matter?
        
    return dict_of_results

def micro_scores(dict_of_results: Dict) -> List:
    """
    This function calculates the average of all tp, fp, fn, tn for all classes and provides a micro precision and recall and F1
    """
    averages = {}
    averages['TP'] = sum(dict_of_results['TP'])
    averages['FP'] = sum(dict_of_results['FP'])
    averages['FN'] = sum(dict_of_results['FN'])
    averages['TN'] = sum(dict_of_results['TN'])
    microPrec = averages['TP']/(averages['TP'] + averages['FP'])
    microRec = averages['TP']/(averages['TP'] + averages['FN'])
    microF1 = (2*microPrec*microRec)/(microPrec+microRec)

    return [microPrec, microRec, microF1]

if __name__ == '__main__':
    dict_of_result = evaluate_predictions(list_of_labels=[2, 2, 3, 4, 8, 6, 1, 8],
                                          list_of_predicted_labels=[2, 5, 8, 1, 8, 4, 4, 6],
                                          list_of_artists=[1, 2, 3, 4, 5, 6, 7, 8])
    print(dict_of_result)
