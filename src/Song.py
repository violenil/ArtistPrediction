from typing import List, Tuple, Dict, Set
import torch
import re
import gensim
import numpy as np
from nltk.tokenize import RegexpTokenizer
import fasttext

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from nltk.stem import PorterStemmer
ps = PorterStemmer()

tokenizer = RegexpTokenizer(r'\w+')


embedding = gensim.models.KeyedVectors.load_word2vec_format('../benchmark/GoogleNews-vectors-negative300.bin', binary='True')
#embedding = fasttext.load_model('../benchmark/cc.en.300.bin')


_UNK_ = np.random.randn(300)
_PAD_STR_ = 'NUTUN_SOBDO'
_PAD_VEC_ = np.zeros(300)


class Song:
    def __init__(self, label: str, song_name: str, lyrics: str, artist_id: int) -> None:
        self.label = label
        self.artist_id = artist_id
        self.song_name = self.get_tokenized_data(song_text=song_name)
        tokenized_lyrics = self.get_tokenized_data(song_text=lyrics)
        self.lyrics = self.resize_lyrics(tokenized_lyrics)
        self.feature_vector = []

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

    def get_feature_vector(self):
        feature_vector_type='embedding'
        if feature_vector_type=='bow_feature':
            return self.bow_feature_extraction()
        elif feature_vector_type=='embedding':
            return self.get_embedding()
        elif feature_vector_type=='manually selected feature':
            pass

if __name__ == '__main__':
    s = Song('The Beatles', 1, 'Hills of green',
             "These are the \nlyrics of this song composed by a group of chaps. I think it's quite nice.")
    print(s.lyrics)
