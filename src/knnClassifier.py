import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from nltk import word_tokenize
from feature_extraction import extract_unique_song_features
from typing import List, Tuple, Dict, Set
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

def filter_content(data:pd.DataFrame,k:int)->pd.DataFrame:
    """
    select only few songs based on some criteria.
    @param data:
    @return:
    """
    dict_of_artist_song_freq = {}
    all_artists = data['artist']
    for n, artist in all_artists.items():
        if artist in dict_of_artist_song_freq.keys():
            dict_of_artist_song_freq[artist] += 1
        else:
            dict_of_artist_song_freq[artist] = 1
    top_k_artists_and_freq=Counter(dict_of_artist_song_freq).most_common(k)
    top_k_artists=[artist for artist,freq in top_k_artists_and_freq]
    reduced_data=data.loc[data['artist'].isin(top_k_artists)]
    return reduced_data

df = pd.read_csv("ArtistPrediction/benchmark/songdata.csv")
reduced_df = filter_content(df, 50)  #only want the 441 artists with most no of songs (>50)
lyrics = reduced_df['text']
artists = reduced_df['artist']

tokenized_lyrics = []
for song in lyrics:
    tokenized_lyrics.append(word_tokenize(song.lower()))

lyrics = lyrics.values.tolist()

def extract_unique_song_features(tokens: List, nouns: List, functionWords: List, wordAssociations: Dict, allEmotions: List, tfidf_transformer, vectorizer, bi_tfidf_transformer, bi_vectorizer, lyrics:str) -> None:
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
    emotion_feature_vector = feature_extraction.extract_emotions(tokens, wordAssociations, allEmotions)
    longest_word_length_feature = feature_extraction.find_length_of_longest(tokens)
    repetition_rate = feature_extraction.calculate_repetition_rate(tokens)

    tfidf_score = feature_extraction.calculate_tfidf_score(tfidf_transformer, vectorizer, lyrics)
    bi_tfidf_score = feature_extraction.calculate_tfidf_score(bi_tfidf_transformer, bi_vectorizer, lyrics)


    #popular_nouns = ["love", "time", "way", "day", "baby", "heart", "life", "night", "gonna", "man"]
    freq_nouns_count = feature_extraction.count_freq_nouns(tokens, nouns)
    freq_func_count = feature_extraction.count_freq_nouns(tokens, functionWords)
    freq_punct_count = feature_extraction.count_freq_nouns(tokens, [",",".","!","?","'"])

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

"""
Feature Extraction
    - import NRCLexicon, which has word to emotion mappings
    - create list of Plutchik emotions (8) + valence (2)
    - create dictionary of wordAssociations: {word:[emotion, emotion], word:[emotion],...}
    - create feature for each song!
"""

NRC = pd.read_csv('ArtistPrediction/benchmark/NRCLexicon.csv',sep='\t')
newNRC = NRC.groupby(['word']).agg(lambda x: tuple(x)).applymap(list).reset_index() #because words have multiple emotions, want list of emotions per word
wordAssociations = dict(zip(newNRC['word'], newNRC['emotion'])) # this is the database that you want to give to the feature extraction

allEmotions = ['anger', 'fear', 'anticipation', 'trust', 'surprise', 'sadness', 'joy', 'disgust', 'negative', 'positive']
nouns = []
with open("ArtistPrediction/benchmark/50MostFrequentNouns.txt") as f:
    nouns = f.read().splitlines()

functionWords = []
with open("ArtistPrediction/benchmark/50ProPrepDet.txt") as ff:
    functionWords = ff.read().splitlines()

 #unigram tfidf scores
vectorizer = CountVectorizer()
count_vector = vectorizer.fit_transform(lyrics)#count vector for all training documents
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(count_vector) #learns the global idf vector

#bigram tfidf scores
bi_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2,2))
bi_count_vector = bi_vectorizer.fit_transform(lyrics)#also a count vector but with bigrams
bi_tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
bi_tfidf_transformer.fit(bi_count_vector)


dataset = []
for i in tqdm(range(len(tokenized_lyrics)), desc="create feature vector"):
    feature_vector = extract_unique_song_features(tokenized_lyrics[i], nouns, functionWords, wordAssociations, allEmotions, tfidf_transformer, vectorizer, bi_tfidf_transformer, bi_vectorizer, lyrics[i])
    dataset.append(feature_vector)

#X = dataset.iloc[:, :-1].values  #takes first to penultimate - attributes of the instances
#y = dataset.iloc[:, 4].values    #takes only the last column - the labels

X = dataset
y = artists

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  #split dataset into 80% train and 20% test

scaler = StandardScaler()
scaler.fit(X_train) #feature scaling

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors = 10)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
