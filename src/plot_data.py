import numpy as np
import matplotlib.pyplot as plt
from typing import List

def plot_data(list_of_scores:List,yaxis_label:str, file_name:str )->None:
    x=list(range(1,len(list_of_scores)+1))
    y=list_of_scores

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='epoch', ylabel=yaxis_label)
    ax.grid()

    fig.savefig("../Plots/"+file_name+".pdf")
    # plt.show()

