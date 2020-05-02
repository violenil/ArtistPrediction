from typing import List, Dict, Tuple
from Perceptron import Perceptron

class MCP:
    def __init__(self, feature_vector:List, weight_vector:List, no_of_class:int) -> None:
        list_of_perceptrons = []
        for c in range(3):
            obj = Perceptron(feature_vec=[1, 0, 1, 0, 1], weight_vec=[0.2, 0.1, 0, 0, 0.6])
            list_of_perceptrons.append(obj)




if __name__ == '__main__':
    m=MCP(feature_vector= feature_vec, weight_vector= weight_vec, no_of_class=total_class)