from typing import List, Dict, Tuple
import numpy as np


class Perceptron:
    """
    It takes one feature vector and a weight vector as input and returns the score based on the
    feature_vector multiplied by weight. Based on the retrieved scores the weight gets updated.
    """

    def __init__(self, weight_vec: List) -> None:
        self.weight_vec = weight_vec

    def find_score(self, feature_vec: List) -> float:
        """
        Score is calculated by matrix multiplication of feature vector and weight vector

        @param feature_vec:
        @return:
        """
        a = np.array(feature_vec)
        b = np.array(self.weight_vec)
        score = np.matmul(a, b)
        return score

    def update_weight_vec(self, change_in_weight: List) -> None:
        """
        Update the weight of the perceptron

        @param change_in_weight: The vector that contains the amount of change to the weight vector
        @return:
        """
        for i in range(len(change_in_weight)):
            self.weight_vec[i] = self.weight_vec[i] + change_in_weight[i]


if __name__ == '__main__':
    p = Perceptron(weight_vec=[1, 1, 1])
    feature_vec = [[1, 1, 1], [1, 2, 2], [1, 1, 3]]
    label = [1, 1, -1]

    score_list = []
    for epoch in range(3):
        for i in range(len(feature_vec)):
            score = p.find_score(feature_vec=feature_vec[i])
            # for j in range(len(score_list)):
            if label[i] == 1:
                if score > 0:
                    change_in_weight = [j * 0 for j in feature_vec[i]]
                else:
                    change_in_weight = feature_vec[i]
            else:
                if score > 0:
                    change_in_weight = [j * -1 for j in feature_vec[i]]
                else:
                    change_in_weight = [j * 0 for j in feature_vec[i]]
            p.update_weight_vec(change_in_weight=change_in_weight)
            print(p.weight_vec)
            score_list.append(score)
    print(score_list)
