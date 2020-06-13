import torch
import torch.nn as nn
from typing import List


class Classifier(nn.Module):
    def __init__(self, feature_vector_size:int, no_of_labels: int):
        super(Classifier, self).__init__()
        # self.rnn = nn.GRU(input_size=embedding_size, hidden_size=300, batch_first=True,
        # self.feature_vector=feature_vector
        # feature_vector_size=(len(feature_vector))
        self.linear_classifier = nn.Linear(in_features=feature_vector_size, out_features=no_of_labels)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, embedded_input: torch.Tensor) -> torch.Tensor:
        predicted_value = self.linear_classifier(feature_vector)
        predicted_value = self.dropout(predicted_value)
        predicted_probabilities = self.softmax(predicted_value)
        return predicted_probabilities
