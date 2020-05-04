from typing import List, Tuple, Dict

class Song:
    def __init__(self, label: int, lyrics: List, feature_vector: List[int]) -> None:
        self.label = label
        self.lyrics = lyrics
        self.feature_vector = feature_vector

    def bow_feature_extraction(self, vocab: List[str]) -> List:
        """
        this method considers bag of words model for feature extraction
        the vocab for this method needs to be created elsewhere, consisting of all types in our
        song collection
        """
        feat_vec = []
        for word in vocab:
            if word in self.lyrics:
                feat_vec.append(1)
            else:
                feat_vec.append(0)
        assert len(vocab) == len(feat_vec)
        return feat_vec


