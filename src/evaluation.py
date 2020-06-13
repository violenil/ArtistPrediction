import random
from typing import List, Dict, Tuple


def artistPredictor(list_artist_frequency: List) -> List:
    list_of_predicted_labels = []
    for i in range(len(list_artist_frequency)):
        list_of_predicted_labels.append(random.choice(list_artist_frequency))
    return list_of_predicted_labels


def evaluate_predictions(list_of_labels: List, list_of_predicted_labels: List, list_of_artists: List) -> Dict:
    """
    This function takes three lists as parameters.

    :param list_of_labels: this contains all the correct labels of the dataset. E.g. [2,2,3,4,8,6,1,8]
    :param list_of_predicted_labels: this contains all the labels predicted by our classifier. E.g. [2,5,8,1,8,4,4,6]
    :param list_of_artists: The list of all the artists in our dataset. E.g. [1,2,3,4,5,6,7,8]
    :return: a dictionary of TP, FP, FN and TN which looks like {TP:[], FP:[], FN:[], TN:[]}
    """
    #list_acc = []
    dict_of_results = {'TP': [], 'FP': [], 'FN': [], 'TN': []}
    for artist in list_of_artists:  # looping through each class (artist) at a time
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        indexesPred = [i for i, x in enumerate(list_of_predicted_labels) if x == artist]
        indexesGold = [i for i, x in enumerate(list_of_labels) if
                       x == artist]  # retrieving all indexes of the artist in predicted and gold, to compare them
        for index in indexesPred:  # going through all predicted indexes for this artist
            if index in indexesGold:  # if we find a match, then tp is increased by 1
                tp += 1
            else:
                fp += 1  # if it doesnt occur, we have a false positive
        for index in indexesGold:  # going through all gold indexes to see if we missed any
            if index not in indexesPred:  # if they dont occur in predicted index list, then we have a false negative
                fn += 1
            else:
                tn += 1

        dict_of_results['TP'].append(tp)
        dict_of_results['FP'].append(fp)
        dict_of_results['FN'].append(fn)
        dict_of_results['TN'].append(
            tn)  # Note: I haven't given any thought to tn, how to calculate this? Does it matter?
        #acc = (tp + tn) / (tp + fn + fp + tn)
        #acc = 0
        #list_acc.append(acc)
        assert len(dict_of_results['TP']) == len(dict_of_results['FP']) and len(dict_of_results['FP']) == len(
            dict_of_results['FN'])  # just to ensure they are same len
    #mean_acc = sum(list_acc) / len(list_acc)
    #dict_of_results['ACC'] = mean_acc
    return dict_of_results


def micro_scores(dict_of_results: Dict) -> Dict:
    """
    This function calculates the average of all tp, fp, fn, tn for all classes and provides a micro precision and recall and F1
    """
    averages = {}
    averages['TP'] = sum(dict_of_results['TP'])
    averages['FP'] = sum(dict_of_results['FP'])
    averages['FN'] = sum(dict_of_results['FN'])
    averages['TN'] = sum(dict_of_results['TN'])
    microPrec = averages['TP'] / (averages['TP'] + averages['FP'])
    microRec = averages['TP'] / (averages['TP'] + averages['FN'])
    microF1 = (2 * microPrec * microRec) / (microPrec + microRec)

    return {'microPrec': microPrec, 'microRec': microRec, 'microF1': microF1}


def macro_scores(dict_of_results: Dict) -> Dict:
    """
    This function calculates precision and recall for each class, then takes the average precision and recall to calculate F1
    """
    prec_list = []
    rec_list = []
    f1_list = []
    for i in range(len(dict_of_results['TP'])):
        tp = dict_of_results['TP'][i]
        fp = dict_of_results['FP'][i]
        fn = dict_of_results['FN'][i]
        if tp == 0:  # problem of zero nominator
            prec_list.append(0.0)
            rec_list.append(0.0)
            f1_list.append(0.0)
        else:
            curr_prec = tp / (tp + fp)
            curr_rec = tp / (tp + fn)
            prec_list.append(curr_prec)
            rec_list.append(curr_rec)
            f1_list.append((2 * curr_prec * curr_rec) / (curr_prec + curr_rec))
    macroPrec = sum(prec_list) / len(prec_list)
    macroRec = sum(rec_list) / len(rec_list)
    macroF1 = sum(f1_list) / len(f1_list)

    return {'macroPrec': macroPrec, 'macroRec': macroRec, 'macroF1': macroF1}


if __name__ == '__main__':
    dict_of_results = evaluate_predictions(list_of_labels=[1, 1, 1, 0, 1, 1, 1, 2, 2, 0, 0, 0, 1, 2, 1, 0, 2, 2, 2, 1],
                                           list_of_predicted_labels=[1, 0, 1, 0, 1, 1, 2, 2, 0, 2, 2, 2, 1, 0, 2, 0, 2,
                                                                     2, 2, 1],
                                           list_of_artists=[0, 1, 2])
    print(dict_of_results)
    print(micro_scores(dict_of_results))
    print(macro_scores(dict_of_results))
