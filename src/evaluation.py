import pandas as pd
#import string
#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer

from DataSet import Dataset
import random
from typing import List, Dict, Tuple


def artistPredictor(batch:List, list_artist_frequency: List) -> List:
    list_of_predicted_labels = []
    for i in range(len(batch)):
        list_of_predicted_labels.append(random.choice(list_artist_frequency))
    return list_of_predicted_labels


def extract_labels(batch: List) -> List:
    """
    Called from train or test neural network functions and it extracts the labels
    from the respective batch, and appends them to a separate list.
    """
    list_labels = []
    for song in batch:
        output = song[0]
        list_labels.append(output)
    return list_labels




# df = pd.read_csv('../songdata.csv')
# artists = df['artist'].unique()
#
# lyrics = df['text']

# def artistPredictor(lyr, art):
#     """
#         I tried 'random' prediction of artist first, had lower precision and fscore
#         than when you fix the artist prediction to one artist, eg. ABBA (quite a lot of songs)
#     """
#     artistListPred = []
#     for lyric in lyr:
#         artistListPred.append("ABBA")
#         #artistListPred.append(random.choice(art))
#     return artistListPred

# predictions = artistPredictor(lyrics, artists)
# correct = df['artist']

def evaluation(pred, correct):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    index = 0
    for artist in pred:
        if artist == correct[index]:
            tp += 1
        else:
            fn += 1
            fp += 1
        index += 1
    return [tp, fp, fn, tn]

