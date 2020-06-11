from typing import List, Tuple


def evaluation_of_prediction(labels: List, predicted_labels: List) -> Tuple[float, float, float, float]:
    # Evaluate p(buggy) rate
    tp, fp, tn, fn = 0, 0, 0, 0
    for i, j in zip(labels, predicted_labels):
        pbuggy = int(i)
        pred_pbuggy = int(j)
        if pbuggy == 1:
            if pred_pbuggy == 1:
                tp += 1
            else:
                fn += 1
        else:  # pbuggy == 0
            if pred_pbuggy == 1:
                fp += 1
            else:
                tn += 1
    if tp + fp:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        if precision and recall:
            fscore = (2 * precision * recall) / (precision + recall)
        else:
            fscore = 0
        return precision, recall, accuracy, fscore
    else:
        return 0, 0, 0, 0


if __name__ == '__main__':
    labels = [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1,
              1]
    predictions = [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0,
                   0, 1]
    value = evaluation_of_prediction(labels, predictions)
    print(value)
