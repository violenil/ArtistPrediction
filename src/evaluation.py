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


def evaluate_predictions(list_of_labels: List, list_of_predicted_labels: List, list_of_artists: List) -> Dict:
    """
    This function takes three lists as parameters.

    :param list_of_labels: this contains all the correct labels of the dataset. E.g. [2,2,3,4,8,6,1,8]
    :param list_of_predicted_labels: this contains all the labels predicted by our classifier. E.g. [2,5,8,1,8,4,4,6]
    :param list_of_artists: The list of all the artists in our dataset. E.g. [1,2,3,4,5,6,7,8]
    :return: a dictionary of TP, FP, FN and TN which looks like {TP:[], FP:[], FN:[], TN:[]}
    """
    dict_of_results = {'TP': [], 'FP': [], 'FN': [], 'TN': []}
    # TODO: Implement here.
    return dict_of_results


if __name__ == '__main__':
    dict_of_result = evaluate_predictions(list_of_labels=[2, 2, 3, 4, 8, 6, 1, 8],
                                          list_of_predicted_labels=[2, 5, 8, 1, 8, 4, 4, 6],
                                          list_of_artists=[1, 2, 3, 4, 5, 6, 7, 8])
    print(dict_of_result)


# def evaluation(pred, correct):
#     tp = 0
#     tp = 0
#     fp = 0
#     fn = 0
#     tn = 0
#     index = 0
#     for artist in pred:
#         if artist == correct[index]:
#             tp += 1
#         else:
#             fn += 1
#             fp += 1
#         index += 1
#     return [tp, fp, fn, tn]
