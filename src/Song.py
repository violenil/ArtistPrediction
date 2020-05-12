from typing import List, Tuple, Dict, Set


class Song:
    def __init__(self, label: str, artist_id: int, song_name: str, lyrics: str) -> None:
        self.label = label
        self.artist_id = artist_id
        self.song_name = self.get_tokenized_data(song_text=song_name)
        self.lyrics = self.get_tokenized_data(song_text=lyrics)
        self.feature_vector=[]

    def __str__(self):

        return self.label+','+str(self.song_name)+','+str(self.lyrics)+','+str(self.artist_id)

    def get_tokenized_data(self, song_text: str) -> List:
        """
        called from get_data each time while we retrieve data from 'content' and return the tokens of the data.
        """
        # tokens = word_tokenize(song_text)

        tokens = song_text.split(' ')
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
            idx=vocab[word]
            feat_vec.append(idx)
        # assert len(vocab) == len(feat_vec)
        self.feature_vector = feat_vec
