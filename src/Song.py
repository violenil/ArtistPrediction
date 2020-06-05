from typing import List, Tuple, Dict, Set
import torch
import re
import gensim
import numpy as np

embedding = gensim.models.KeyedVectors.load_word2vec_format(
    '../benchmark/GoogleNews-vectors-negative300.bin', binary='True')

_UNK_ = np.random.rand(embedding.vector_size)
_PAD_STR_ = 'NUTUN_SOBDO'
_PAD_VEC_ = [0] * embedding.vector_size


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
        tokens = [match.group(0) for match in
                  re.finditer(r"\w+|([^\w])\1*", song_text)]  # keeps punctuation and \n chars

        # tokens = word_tokenize(song_text)

        # tokens = song_text.split(' ')
        # tokens = get_tokens(text=song_text, chars=[' ', ',', "'"])
        return tokens

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
        if len(tokenized_lyrics) > 75:
            tokenized_lyrics = tokenized_lyrics[:75]
        while len(tokenized_lyrics) < 75:
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
        return torch.FloatTensor(list_with_embeddings)


if __name__ == '__main__':
    s = Song('The Beatles', 1, 'Hills of green',
             "These are the \nlyrics of this song composed by a group of chaps. I think it's quite nice.")
    print(s.lyrics)
