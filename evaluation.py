import pandas as pd
#import string
#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer
import random

df = pd.read_csv('./songdata.csv')
artists = df['artist'].unique()

lyrics = df['text']

def artistPredictor(lyr, art):   
    """
        I tried 'random' prediction of artist first, had lower precision and fscore
        than when you fix the artist prediction to one artist, eg. ABBA (quite a lot of songs)
    """
    artistListPred = []
    for lyric in lyr:
        artistListPred.append("ABBA")
        #artistListPred.append(random.choice(art))
    return artistListPred

predictions = artistPredictor(lyrics, artists)
correct = df['artist']

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

evaluat = evaluation(predictions, correct)

p = evaluat[0]/(evaluat[0]+evaluat[1])
r = evaluat[0]/(evaluat[0]+evaluat[3])

fscore = 2*((p*r)/(p+r))

print("Precision: " + str(p))
print("Recall: " + str(r))
print("F-score: " + str(fscore))
