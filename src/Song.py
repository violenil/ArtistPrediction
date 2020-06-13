from typing import List, Tuple, Dict, Set
import torch
import re
import gensim
import numpy as np
import pandas as pd

from nltk import word_tokenize, pos_tag
import feature_extraction
import fasttext

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

from nltk.stem import PorterStemmer

ps = PorterStemmer()
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

embedding = gensim.models.KeyedVectors.load_word2vec_format('../benchmark/GoogleNews-vectors-negative300.bin',
                                                            binary='True')
# embedding = fasttext.load_model('../benchmark/cc.en.300.bin')


_UNK_ = np.random.randn(300)
_PAD_STR_ = 'NUTUN_SOBDO'
_PAD_VEC_ = np.zeros(300)

"""
Feature Extraction
    - import NRCLexicon, which has word to emotion mappings
    - create list of Plutchik emotions (8) + valence (2)
    - create dictionary of wordAssociations: {word:[emotion, emotion], word:[emotion],...}
    - create feature for each song!
"""

NRC = pd.read_csv('../benchmark/NRCLexicon.csv', sep='\t')
newNRC = NRC.groupby(['word']).agg(lambda x: tuple(x)).applymap(
    list).reset_index()  # because words have multiple emotions, want list of emotions per word
wordAssociations = dict(zip(newNRC['word'], newNRC[
    'emotion']))  # this is the database that you want to give to the feature extraction

allEmotions = ['anger', 'fear', 'anticipation', 'trust', 'surprise', 'sadness', 'joy', 'disgust',
               'negative', 'positive']
nouns = []
with open("../benchmark/50MostFrequentNouns.txt") as f:
    nouns = f.read().splitlines()

functionWords = []
with open("../benchmark/50ProPrepDet.txt") as ff:
    functionWords = ff.read().splitlines()


class Song:
    def __init__(self, label: str, song_name: str, lyrics: str, artist_id: int) -> None:
        self.label = label
        self.artist_id = artist_id
        self.song_name = word_tokenize(song_name)
        tokenized_lyrics = word_tokenize(lyrics.lower())
        self.lyrics = self.resize_lyrics(tokenized_lyrics)

    def __str__(self) -> str:
        return self.label + ',' + str(self.song_name) + ',' + str(self.lyrics) + ',' + str(self.artist_id)

    def get_tokenized_data(self, song_text: str) -> List:
        """
        called from get_data each time while we retrieve data from 'content' and return the tokens of the data.
        """
        # tokens = [match.group(0) for match in re.finditer(r"\w+|([^\w])\1*", song_text)]  # keeps punctuation and \n chars
        tokens = tokenizer.tokenize(song_text)
        stemmed_words = self.get_stem_data(tokens)
        lyrics_without_stop_words = self.del_stopword(stemmed_words)
        return lyrics_without_stop_words

    def get_stem_data(self, list_words):
        list_stemmed_words = []
        for word in list_words:
            list_stemmed_words.append(ps.stem(word))
        return list_stemmed_words

    def del_stopword(self, list_words):
        filtered_text = []
        for word in list_words:
            if word not in stop_words:
                filtered_text.append(word)
        return filtered_text

    def extract_unique_song_features(self, nouns: List, functionWords: List, wordAssociations: Dict,
                                     allEmotions: List) -> None:
        """
        This method extracts a set of features:
            8 emotions (anger, fear, anticipation, trust, surprise, sadness, joy, disgust)
            2 polarities (negative, positive)
            length of longest word in song
            repetition rate (unique_words/total_words)
            count of \n chars
            50 most frequent nouns in songs generally (love, time, way, day, ...) --> freq count
                (could also do binary for the above)
            50 most frequent function words (contains pronouns like eg. "I" and "you", determiners eg. "the" and "a" and prepositions eg. "in")
            5 punctuation symbols (',','.','!','?',''') --> same method as above
            1 count for song name length
            1 feature set to 1 for all songs
        """
        feat_vec = []
        emotion_feature_vector = feature_extraction.extract_emotions(self.lyrics, wordAssociations, allEmotions)
        longest_word_length_feature = feature_extraction.find_length_of_longest(self.lyrics)
        repetition_rate = feature_extraction.calculate_repetition_rate(self.lyrics)

        freq_nouns_count = feature_extraction.count_freq_nouns(self.lyrics, nouns)
        freq_func_count = feature_extraction.count_freq_nouns(self.lyrics, functionWords)
        freq_punct_count = feature_extraction.count_freq_nouns(self.lyrics, [",", ".", "!", "?", "'"])

        feat_vec += emotion_feature_vector
        feat_vec.append(longest_word_length_feature)
        feat_vec.append(repetition_rate)
        feat_vec.append(self.lyrics.count("\n"))  # count all \n chars, didn't need a method for that
        feat_vec += freq_nouns_count
        feat_vec += freq_func_count
        feat_vec += freq_punct_count
        feat_vec.append(len(self.song_name))
        feat_vec.append(1)
        return feat_vec

    def resize_lyrics(self, tokenized_lyrics: List) -> List:
        token_length = 50
        if len(tokenized_lyrics) > token_length:
            tokenized_lyrics = tokenized_lyrics[:token_length]
        while len(tokenized_lyrics) < token_length:
            tokenized_lyrics.append(_PAD_STR_)
        return tokenized_lyrics

    def get_embedding(self) -> torch.Tensor:
        list_with_embeddings = []
        for word in self.lyrics:
            if word == _PAD_STR_:
                list_with_embeddings.append(_PAD_VEC_)
            elif word not in embedding:
                list_with_embeddings.append(_UNK_)
            else:
                list_with_embeddings.append(embedding[word])
        return list_with_embeddings

    def get_feature_vector(self, feature_vector_type):
        if feature_vector_type == 'manual_features':
            return self.extract_unique_song_features(nouns, functionWords, wordAssociations, allEmotions)
        elif feature_vector_type == 'RNN':
            return self.get_embedding()


if __name__ == '__main__':
    s = Song('The Beatles', 1, 'Hills of green',
             "These are the \nlyrics of this song composed by a group of chaps. I think it's quite nice.")
    print(s.lyrics)
