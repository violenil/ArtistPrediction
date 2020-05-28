from typing import List, Tuple, Dict, Set
from nltk import word_tokenize, pos_tag
import re


class Song:
    def __init__(self, label: str, artist_id: int, song_name: str, lyrics: str) -> None:
        self.label = label
        self.artist_id = artist_id
        self.song_name = self.word_tokenize(song_text=song_name)
        shortend_lyrics = self.word_tokenize(song_text=lyrics)
        self.lyrics = shortend_lyrics
        tagged_lyrics = pos_tag(shortend_lyrics) #returns list of tuples [(word, tag), ...]
        self.tagged_lyrics = dict((y, x) for x, y in tagged_lyrics) # makes into dict {tag:word, tag:word, ...}
        self.feature_vector = []

    def __str__(self) -> str:

        return self.label + ',' + str(self.song_name) + ',' + str(self.lyrics) + ',' + str(self.artist_id)

    def get_tokenized_data(self, song_text: str) -> List:
        """
        called from get_data each time while we retrieve data from 'content' and return the tokens of the data.
        """
        tokens = [match.group(0) for match in
                  re.finditer(r"\w+|([^\w])\1*", song_text)]  # keeps punctuation and \n chars


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
    
    def extract_emotions(self, 

    def extract_unique_song_features(self, nouns: Dict) -> None:
        """
        This method extracts a set of features:
            8 emotions (anger, fear, anticipation, trust, surprise, sadness, joy, disgust)
            2 polarities (negative, positive)
            length of longest word in song
            repetition rate (unique_words/total_words)
            count of \n chars
            10 most frequent nouns in songs generally (love, time, way, day, baby, heart, life, night, gonna, man) --> binary
            counts for 5 punctuation symbols (',','.','!','?',''')
        """
        feat_vec = []
        # TO DO    

if __name__ == '__main__':
    s = Song('The Beatles', 1, 'Hills of green',
             "These are the \nlyrics of this song composed by a group of chaps. I think it's quite nice.")
    print(s.lyrics)
