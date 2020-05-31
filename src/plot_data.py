import json
import matplotlib.pyplot as plt
from typing import List

def plot_data(list_of_scores:List,yaxis_label:str, file_name:str )->None:
    x=list(range(1,len(list_of_scores)+1))
    y=list_of_scores

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='epoch', ylabel=yaxis_label)
    ax.grid()

    fig.savefig("../otherPlots/"+file_name+".pdf")
    # plt.show()
if __name__ == '__main__':
    with open('../results/2_artists_1000.json') as j:
        data=json.load(j)
    plot_data(list_of_scores=data['macro_f_score'], yaxis_label='f_score',
              file_name=str(data['no_of_artists']) + '_artists_' + str(data['no_of_epochs']))
