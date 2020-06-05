import torch
import torch.nn as nn
from typing import List


class Classifier:
    def __init__(self, embedding_size:int, no_of_labels:int):
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=100, batch_first=True)
        self.linear_classifier = nn.Linear(in_features=100,out_features=no_of_labels)
        self.softmax= nn.Softmax(dim=1)
        """
        why dim =1?
        
        """

    def forward(self, embedded_input: torch.Tensor) -> torch.Tensor:
        output, hidden_layer = self.rnn(embedded_input)
        predicted_value = self.linear_classifier(hidden_layer)
        predicted_probs = self.softmax(predicted_value)
        return predicted_probs
