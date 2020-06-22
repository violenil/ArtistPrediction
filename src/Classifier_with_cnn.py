import torch
import torch.nn as nn
from typing import List


class Classifier_using_cnn(nn.Module):
    def __init__(self, embedding_size: int, no_of_labels: int):
        super(Classifier_using_cnn, self).__init__()
        self.name = 'using_cnn'
        self.convNet = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, out_channels=400,
                      kernel_size=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(200)
        )
        classifier_in_feature_size = 600
        self.first_linear_layer = nn.Linear(in_features=400, out_features=classifier_in_feature_size)
        self.second_linear_classifier = nn.Linear(in_features=classifier_in_feature_size, out_features=no_of_labels)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)
        self.loss_calc = nn.NLLLoss()

    def forward(self, embedded_input: torch.Tensor) -> torch.Tensor:
        embedded_input = embedded_input.permute(0, 2, 1)
        output = self.convNet(embedded_input)
        output = torch.squeeze(output)
        first_predicted_value = self.first_linear_layer(output)
        final_predicted_value = self.second_linear_classifier(first_predicted_value)
        predicted_value = self.dropout(final_predicted_value)
        predicted_probabilities = self.softmax(predicted_value)
        return predicted_probabilities
