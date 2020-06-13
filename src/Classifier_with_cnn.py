import torch
import torch.nn as nn
from typing import List


class Classifier_using_cnn(nn.Module):
    def __init__(self, embedding_size: int, no_of_labels: int):
        super(Classifier_using_cnn, self).__init__()
        self.name = 'using_cnn'
        self.conv=nn.Conv1d(
            in_channels=embedding_size,out_channels=400,kernel_size=,stride=1
        )
        classifier_in_feature_size=600
        self.first_linear_layer = nn.Linear(in_features=400, out_features=classifier_in_feature_size)
        self.second_linear_classifier= nn.Linear(in_features=classifier_in_feature_size, out_features=no_of_labels)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)
        self.loss_calc = nn.NLLLoss()

    def forward(self, embedded_input: torch.Tensor) -> torch.Tensor:
        output=self.conv(embedded_input)
        first_predicted_value = self.first_linear_layer(output)
        final_predicted_value=self.self.second_linear_classifier(first_predicted_value)
        predicted_value = self.dropout(final_predicted_value)
        predicted_probabilities = self.softmax(predicted_value)
        return predicted_probabilities
