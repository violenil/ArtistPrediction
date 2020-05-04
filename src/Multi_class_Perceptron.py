from typing import List, Dict, Tuple
from Perceptron import Perceptron


class MCP:
    def __init__(self, feature_vector: List, weight_vector: List, classes: List[str]) -> None:
        self.dict_of_perceptrons = {}
        for cls in classes:
            obj = Perceptron(feature_vec=[1, 0, 1, 0, 1], weight_vec=[0.2, 0.1, 0, 0, 0.6])
            self.dict_of_perceptrons[cls] = obj

    def find_all_scores(self) -> Dict:
        dict_of_scores = {}
        for perceptron in self.dict_of_perceptrons:
            dict_of_scores[perceptron] = self.dict_of_perceptrons[perceptron].find_score()
        return dict_of_scores

    def find_max(self, dict_of_scores: Dict) -> Tuple[str, float]:
        """
        TODO: Find max score. Return the key and score in the form of tuple.
        """
        return predicted_label, predicted_score

    def update_weight(self, actual_label, predicted_label, actual_label_score, predicted_label_score) -> None:
        """
        TODO: write code to find weight and update weight for the respective two classes.
        """
        pass


if __name__ == '__main__':
    m = MCP(feature_vector=feature_vec, weight_vector=weight_vec, classes=total_class)
    for i in range(5):
        dict_of_scores = m.find_all_scores()
        predicted_label, max_score = m.find_max(dict_of_scores=dict_of_scores)
        m.update_weight(actual_label=actual_label, predicted_label=predicted_label,
                        actual_label_score=actual_label_score, max_score=predicted_label_score)
