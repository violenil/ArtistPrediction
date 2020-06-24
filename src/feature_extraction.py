from typing import List, Tuple, Dict, Set
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd

"""
Here are all the functions for extracting features. 'bow_feature_extraction' is also here.
"""


def bow_feature_extraction(self, vocab: Dict) -> None:
    """
    this method considers bag of words model for feature extraction
    the vocab for this method needs to be created elsewhere, consisting of all types in our
    song collection
    """
    feat_vec = []
    for word in self.lyrics:
        if word in vocab:
            idx = vocab[word]
            feat_vec.append(idx)
    # assert len(vocab) == len(feat_vec)
    self.feature_vector = feat_vec


def extract_emotions(lyrics: List, wordAssociations: Dict, allEmotions: List) -> List:
    """
    first find all emotions that come up for the words in your lyrics (may want to exclude function words here using pos)
    then count occurrence of all 8 emotions and neg and positive
    return these as a list of counts of emotions in the given order --> list of length 10
    """
    # reducedTokenTagDict = {word:tag for word, tag in self.tagged_lyrics.items() if tag in ['NN', 'NNP', 'NNS', 'PRP', 'RB', 'JJ', 'JJS', 'VB', 'VBD', 'VBG']}
    # reducedLyrics = list(reducedTokenTagDict.keys())
    # I wanted to only process those words that have a particular tag, but the dict doesnt keep duplicates of words so this is not possible
    # TO DO

    observed_emotions = []  # instances of all associated emotions

    for word in lyrics:
        if word in wordAssociations.keys():
            observed_emotions += wordAssociations[word]

    emotion_feature_vector = []
    for e in allEmotions:
        emotion_feature_vector.append(observed_emotions.count(e))
    assert len(emotion_feature_vector) == 10
    return (emotion_feature_vector)


def find_length_of_longest(lyrics: List):
    """
    search list for longest word and return its length
    """
    # lyrics_list = list(self.tagged_lyrics.keys())
    return (len(max(lyrics, key=len)))


def calculate_repetition_rate(lyrics: List):
    """
    repetition_rate would be the number of unique words in the lyrics divided by the total
    number of words in the lyrics
    """
    # lyrics_list = list(self.tagged_lyrics.keys())
    return (int((len(set(lyrics)) / len(lyrics)) * 10))


def count_freq_nouns(lyrics: List, popNouns: List) -> List:
    """
    takes a list of popular nouns in songs (10 of them) and counts how frequently they occur in
    the lyrics, returning the list of counts
    """
    popCount = []
    for w in popNouns:
        popCount.append(lyrics.count(w))

    return(popCount)

def calculate_tfidf_score(tfidf_transformer, vectorizer, lyrics: str):
    doc_count_vector = vectorizer.transform([lyrics]) #this is a count vector for this particular document
    tfidf_vector = tfidf_transformer.transform(doc_count_vector) #global tf-idf scores for all ngrams in this doc
    feature_names = vectorizer.get_feature_names()
    df = pd.DataFrame(tfidf_vector.T.todense(), index=feature_names, columns=["tfidf"])
    reduced_df = df.sort_values(by=['tfidf'], ascending=False).head(50)
    tfidf_score = round(reduced_df['tfidf'].mean()*10, 3)
    return tfidf_score

