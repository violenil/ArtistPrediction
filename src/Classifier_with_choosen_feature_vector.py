import torch
import torch.nn as nn
from typing import List


class Classifier_manual_features(nn.Module):
    def __init__(self, embedding_size:int, no_of_labels: int):
        super(Classifier_manual_features, self).__init__()
        self.name='using_manual_feature'
        self.linear_classifier = nn.Linear(in_features=embedding_size, out_features=no_of_labels)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)
        self.loss_calc = nn.NLLLoss()


    def forward(self, embedded_input: torch.Tensor) -> torch.Tensor:
        predicted_value = self.linear_classifier(embedded_input)
        predicted_value = self.dropout(predicted_value)
        predicted_probabilities = self.softmax(predicted_value)
        return predicted_probabilities
