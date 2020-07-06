import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from nltk import word_tokenize
import feature_extraction
from data_reconstruction import filter_content
from typing import List, Tuple, Dict, Set
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm


df = pd.read_csv("../benchmark/songdata.csv")
reduced_df = filter_content(df, 5)  #only want the top artists with most no of songs
lyrics = reduced_df['text']
artists = reduced_df['artist']
feat_type = ["Bag of Words", "Hand-Crafted"][1]  #set this for different feature vector

tokenized_lyrics = []
for song in lyrics:
    tokenized_lyrics.append(word_tokenize(song.lower()))

lyrics = lyrics.values.tolist()

'''
create feature vector
'''
 #unigram tfidf scores
vectorizer = CountVectorizer()
count_vector = vectorizer.fit_transform(lyrics)#count vector for all training documents
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(count_vector) #learns the global idf vector

#bigram tfidf scores
#bi_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2,2))
#bi_count_vector = bi_vectorizer.fit_transform(lyrics)#also a count vector but with bigrams
#bi_tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
#bi_tfidf_transformer.fit(bi_count_vector)

vocab = dict()
idx = 0
for song in tokenized_lyrics:
    for w in song:
        if w not in vocab.keys():
            vocab[w] = idx
            idx += 1
        else:
            pass

dataset = []
if feat_type == "Bag of Words":
    for i in tqdm(range(len(tokenized_lyrics)), desc="create feature vector"):
        feature_vector = feature_extraction.create_bow_feature_vector(tokenized_lyrics[i], vocab)
        dataset.append(feature_vector)
elif feat_type == "Hand-Crafted":
    for i in tqdm(range(len(tokenized_lyrics)), desc="create feature vector"):
        feature_vector = feature_extraction.create_manual_feature_vector(tfidf_transformer, vectorizer, tokenized_lyrics[i], lyrics[i])
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
