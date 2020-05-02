from typing import List, Dict, Tuple


class Perceptron:
    """
    we create one perceptron per class. It takes one song and returns the score based on the
    input multiplied by weight. We set a random weight first and learn by and by.
    We create a dictionary of feature scores and add all the relevant values to a get score
    """

    def __init__(self, feature_vec: List, weight_vec: List) -> None:
        self.weight_vec = weight_vec
        self.feature_vec = feature_vec

    def find_score(self) -> float:
        """
        TODO: Multiply weight_vector with feature_vector to find score
        """
        score = 0.0
        return score

    def update_weight_vec(self, label: int, score: float) -> None:
        """
        TODO: slide no:8 03-perc-flat.pdf
        TODO: See slide 10 after implementing the above.
        """
        pass


if __name__ == '__main__':
    p = Perceptron(feature_vec=[1, 0, 1, 0, 1], weight_vec=[0.2, 0.1, 0, 0, 0.6])
    pass
