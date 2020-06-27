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
    #reducedTokenTagDict = {word:tag for word, tag in self.tagged_lyrics.items() if tag in ['NN', 'NNP', 'NNS', 'PRP', 'RB', 'JJ', 'JJS', 'VB', 'VBD', 'VBG']}
    #reducedLyrics = list(reducedTokenTagDict.keys())
    #I wanted to only process those words that have a particular tag, but the dict doesnt keep duplicates of words so this is not possible
    #TO DO

    observed_emotions = [] # instances of all associated emotions

    for word in lyrics:
        if word in wordAssociations.keys():
            observed_emotions += wordAssociations[word]

    emotion_feature_vector = []
    for e in allEmotions:
        emotion_feature_vector.append(observed_emotions.count(e))
    assert len(emotion_feature_vector) == 10
    return(emotion_feature_vector)

def find_length_of_longest(lyrics: List):
    """
    search list for longest word and return its length
    """
    #lyrics_list = list(self.tagged_lyrics.keys())
    return(len(max(lyrics, key=len)))

def calculate_repetition_rate(lyrics: List):
    """
    repetition_rate would be the number of unique words in the lyrics divided by the total
    number of words in the lyrics
    """
    #lyrics_list = list(self.tagged_lyrics.keys())
    return(int((len(set(lyrics))/len(lyrics))*10))

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


def extract_unique_song_features(tfidf_transformer, vectorizer, bi_tfidf_transformer, bi_vectorizer, tokens: List, nouns: List, functionWords: List, wordAssociations: Dict, allEmotions: List, lyrics:str) -> List:
    """
    This method extracts a set of features:
        8 emotions (anger, fear, anticipation, trust, surprise, sadness, joy, disgust)
        2 polarities (negative, positive)
        length of longest word in song
        repetition rate (unique_words/total_words)
        count of \n chars
        50 most frequent nouns in songs generally (love, time, way, day, baby, heart, life, night, gonna, man) --> freq count
            (could also do binary for the above)
        50 most frequent function words (contains pronouns like eg. "I" and "you", determiners eg. "the" and "a" and prepositions eg. "in"
        counts for 5 punctuation symbols (',','.','!','?',''') --> I added this to the above for now (same method)
    """
    feat_vec = []
    emotion_feature_vector = extract_emotions(tokens, wordAssociations, allEmotions)
    longest_word_length_feature = find_length_of_longest(tokens)
    repetition_rate = calculate_repetition_rate(tokens)

    tfidf_score = calculate_tfidf_score(tfidf_transformer, vectorizer, lyrics)
    bi_tfidf_score = calculate_tfidf_score(bi_tfidf_transformer, bi_vectorizer, lyrics)


    #popular_nouns = ["love", "time", "way", "day", "baby", "heart", "life", "night", "gonna", "man"]
    freq_nouns_count = count_freq_nouns(tokens, nouns)
    freq_func_count = count_freq_nouns(tokens, functionWords)
    freq_punct_count = count_freq_nouns(tokens, [",",".","!","?","'"])

    feat_vec += emotion_feature_vector
    feat_vec.append(longest_word_length_feature)
    feat_vec.append(repetition_rate)
    feat_vec.append(tokens.count("\n")) #count all \n chars, didn't need a method for that
    feat_vec += freq_nouns_count
    feat_vec += freq_func_count
    feat_vec += freq_punct_count
    feat_vec.append(tfidf_score)
    feat_vec.append(bi_tfidf_score)

    return feat_vec

def create_feature_vector(tfidf_transformer, vectorizer, bi_tfidf_transformer, bi_vectorizer, tokens: List, lyrics: str):
    """
    Feature Extraction
        - import NRCLexicon, which has word to emotion mappings
        - create list of Plutchik emotions (8) + valence (2)
        - create dictionary of wordAssociations: {word:[emotion, emotion], word:[emotion],...}
        - create feature for each song!
    """

    NRC = pd.read_csv('../benchmark/NRCLexicon.csv',sep='\t')
    newNRC = NRC.groupby(['word']).agg(lambda x: tuple(x)).applymap(list).reset_index() #because words have multiple emotions, want list of emotions per word
    wordAssociations = dict(zip(newNRC['word'], newNRC['emotion'])) # this is the database that you want to give to the feature extraction

    allEmotions = ['anger', 'fear', 'anticipation', 'trust', 'surprise', 'sadness', 'joy', 'disgust', 'negative', 'positive']
    nouns = []
    with open("../benchmark/50MostFrequentNouns.txt") as f:
        nouns = f.read().splitlines()

    functionWords = []
    with open("../benchmark/50ProPrepDet.txt") as ff:
        functionWords = ff.read().splitlines()

    feature_vector = extract_unique_song_features(tfidf_transformer, vectorizer, bi_tfidf_transformer, bi_vectorizer, tokens, nouns, functionWords, wordAssociations, allEmotions, lyrics)
    return feature_vector
