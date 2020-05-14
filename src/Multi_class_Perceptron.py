from typing import List, Dict, Tuple
from Perceptron import Perceptron
import random
import sys
from tqdm import tqdm

random.seed(33)


class MCP:
    """
    Instances per class is created. Scores returned from the classes are compared and the change in weight is calculated.
    The change in weight is then passed to the respective two Perceptrons, where their weights gets updated.
    """

    def __init__(self, classes: List, weight_vec_length: int) -> None:
        """
        One Perceptron per class is created. A random weight vector is passed as argument to each Perceptron.
        Each instances when created are kept in a dictionary, where the keys are the Perceptron names and
        the instances are the values.

        Example: dict_of_perceptron = {'ABBA': obj1, 'Coldplay': obj2}

        @param classes: contains a list of all unique classes.
        @param weight_vec_length:
        """
        self.dict_of_perceptrons = {}
        for perceptron in classes:
            weight_vec = self.create_random_vec(length=weight_vec_length)
            # print(weight_vec)
            self.dict_of_perceptrons[perceptron] = Perceptron(weight_vec=weight_vec)

    def create_random_vec(self, length: int) -> List[int]:
        """
        A random vector of input length gets created.

        @param length:
        @return:
        """
        random_list = random.sample(range(0, length * 5), length)
        return random_list

    def find_all_scores(self, feature_vec: List) -> Dict:
        """
        Takes a feature vector as input and passed to each Perceptron. The scores returned from the Perceptrons
        are kept in a dictionary, where the keys are the Perceptron names and the values are the scores.

        Example: dict_of_scores = {'ABBA': 3, 'Coldplay': -14}

        @param feature_vec:
        @return:
        """
        dict_of_scores = {}
        for perceptron in self.dict_of_perceptrons:
            dict_of_scores[perceptron] = self.dict_of_perceptrons[perceptron].find_score(feature_vec=feature_vec)
        return dict_of_scores

    def find_max(self, dict_of_scores: Dict) -> Tuple[str, float]:
        """
        Takes a dictionary of scores and returns the key and value where the score is maximum.

        @param dict_of_scores:
        @return:
        """
        mx_score = -sys.maxsize  # Initializing max value to -infinity
        mx_perceptron = ''
        for perceptron, score in dict_of_scores.items():
            if score > mx_score:
                mx_score = score
                mx_perceptron = perceptron
        return mx_perceptron, mx_score

    def update_weight(self, actual_perceptron: str, predicted_perceptron: str,
                      feature_vec: List) -> None:
        """
        Takes the Perceptron that has maximum score and the Perceptron that matches with the correct label.

        @param actual_perceptron: The Perceptron that matches the correct label.
        @param predicted_perceptron: The Perceptron with maximum score.
        @param feature_vec:
        @return:
        """
        change_in_wt_for_actual_perceptron = [j * 0 for j in feature_vec]
        change_in_wt_for_predicted_perceptron = [j * 0 for j in feature_vec]
        if predicted_perceptron != actual_perceptron:
            change_in_wt_for_actual_perceptron = feature_vec
            change_in_wt_for_predicted_perceptron = [j * -1 for j in feature_vec]
        # print(change_in_wt_for_actual_perceptron, '\n', change_in_wt_for_predicted_perceptron)
        self.dict_of_perceptrons[actual_perceptron].update_weight_vec(
            change_in_weight=change_in_wt_for_actual_perceptron)
        self.dict_of_perceptrons[predicted_perceptron].update_weight_vec(
            change_in_weight=change_in_wt_for_predicted_perceptron)


if __name__ == '__main__':
    feature_vec = [[1, 1, 1], [1, 2, 2], [1, 1, 3]]
    label = ['positive', 'positive', 'negative']
    total_classes = ['positive', 'negative']
    m = MCP(classes=total_classes, weight_vec_length=len(feature_vec[0]))

    # dict_of_scores={}
    for epoch in range(4):
        for i in range(len(feature_vec)):
            # for i in range(len(total_classes)):
            dict_of_scores = m.find_all_scores(feature_vec[i])
            predicted_perceptron, max_score = m.find_max(dict_of_scores=dict_of_scores)
            actual_perceptron = label[i]
            m.update_weight(actual_perceptron=actual_perceptron, predicted_perceptron=predicted_perceptron,
                            feature_vec=feature_vec[i])
