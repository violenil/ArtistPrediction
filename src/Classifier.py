import torch
import torch.nn as nn
from typing import List


class Classifier(nn.Module):
    def __init__(self, embedding_size: int, no_of_labels: int):
        super(Classifier, self).__init__()
        self.bidirectional_RNN = True
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=300, batch_first=True,
                          bidirectional=self.bidirectional_RNN)
        n = 1
        if self.bidirectional_RNN:
            n = 2
        self.linear_classifier = nn.Linear(in_features=300 * n, out_features=no_of_labels)
        self.softmax = nn.LogSoftmax(dim=1)
        """
        why dim =1?
        
        We calculate the softmax on dim=1 since the output of the Linear layer
        is of shape (batch_size,2)
        Eg. if batch_size = 3 then the Linear layer will output
        will be h = [[1, 2],
            ...      [3, 4],
            ...      [5, 4]]
        We want to calculate the softmax along each row.
        h.shape → (3,2)

        The 0th dim is the batch_size.
        The 1st dim is the actual output.
        For the above example:
        * if dim=1 the softmax will be calculated on each row:
            [[0.2689, 0.7311],
             [0.2689, 0.7311],
             [0.7311, 0.2689]] # See, 5 is greater than 4 and has higher probability
        * if dim=0 the softmax will be calculated on each column:
            [[0.0159, 0.0634],
             [0.1173, 0.4683],
             [0.8668, 0.4683]]
        """
        self.dropout = nn.Dropout(p=0.5)
        self.loss_calc = nn.NLLLoss()

    def forward(self, embedded_input: torch.Tensor) -> torch.Tensor:
        output, hidden_size = self.rnn(embedded_input)
        if self.bidirectional_RNN == True:
            hidden_size = torch.cat((hidden_size[0], hidden_size[1]), 1)
        else:
            hidden_size = torch.squeeze(hidden_size)  # to change the dimension of hidden layer from[1,3,600] to [3,600]
        predicted_value = self.linear_classifier(hidden_size)
        predicted_value = self.dropout(predicted_value)
        predicted_probabilities = self.softmax(predicted_value)
        return predicted_probabilities
