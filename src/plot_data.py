import json
import matplotlib.pyplot as plt
from typing import List


def plot_data(list_of_scores: List, yaxis_label: str, file_name: str) -> None:
    x = list(range(1, len(list_of_scores) + 1))
    y = list_of_scores

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='epoch', ylabel=yaxis_label)
    ax.grid()

    #fig.savefig("../Plots/" + file_name + ".pdf")
    # plt.show()
    fig.savefig("../Plots/" + file_name + ".pdf")


def plot_trainig_validation_loss(training_loss, validation_loss):
    fig, ax = plt.subplots()
    x = list(range(1, len(training_loss) + 1))

    ax.plot(x, training_loss, label='training')
    ax.plot(x, validation_loss, label='validation')

    ax.grid()
    ax.legend()
    ax.set(xlabel='epoch', ylabel='loss')

    # plt.show()
    fig.savefig("../Plots/" + 'training_validation_loss' + ".pdf")


if __name__ == '__main__':
    with open('../results/2_artists_1000.json') as j:
        data=json.load(j)
    plot_data(list_of_scores=data['macro_f_score'], yaxis_label='f_score',
              file_name=str(data['no_of_artists']) + '_artists_' + str(data['no_of_epochs']))
    plot_trainig_validation_loss([2, 3, 1, 4, 5], [5, 2, 1, 3, 5])
