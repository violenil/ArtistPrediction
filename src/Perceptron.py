from typing import List, Dict, Tuple
import numpy as np


class Perceptron:
    """
    we create one perceptron per class. It takes one song and returns the score based on the
    input multiplied by weight. We set a random weight first and learn by and by.
    We create a dictionary of feature scores and add all the relevant values to a get score
    """

    def __init__(self, weight_vec: List) -> None:
        self.weight_vec = weight_vec

    def find_score(self, feature_vec: List) -> float:
        """
        TODO: Multiply weight_vector with feature_vector to find score
        """
        a = np.array(feature_vec)
        b = np.array(self.weight_vec)
        score = np.matmul(a, b)
        return score

    def update_weight_vec(self, change_in_weight: List) -> None:
        """
        TODO: slide no:8 03-perc-flat.pdf
        TODO: See slide 10 after implementing the above.
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
            #for j in range(len(score_list)):
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


