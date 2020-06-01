from typing import List, Tuple, Dict, Set
from nltk import word_tokenize, pos_tag
import re


class Song:
    def __init__(self, label: str, artist_id: int, song_name: str, lyrics: str) -> None:
        self.label = label
        self.artist_id = artist_id
        self.song_name = word_tokenize(song_name)
        shortend_lyrics = word_tokenize(lyrics)
        self.lyrics = shortend_lyrics
        #tagged_lyrics = pos_tag(shortend_lyrics) #returns list of tuples [(word, tag), ...]
        #self.tagged_lyrics = dict((x, y) for x, y in tagged_lyrics) # makes into dict {word:tag, word:tag, ...}

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

if __name__ == '__main__':
    s = Song('The Beatles', 1, 'Hills of green',
             "These are the \nlyrics of this song composed by a group of chaps. I think it's quite nice.")
    print(s.lyrics)
