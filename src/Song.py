from typing import List, Tuple, Dict, Set
from nltk import word_tokenize, pos_tag
import feature_extraction
import re
from nltk.stem.snowball import SnowballStemmer

class Song:
    def __init__(self, label: str, artist_id: int, song_name: str, lyrics: str) -> None:
        self.label = label
        self.artist_id = artist_id
        self.song_name = word_tokenize(song_name)
        shortend_lyrics = word_tokenize(lyrics.lower())
        self.lyrics = shortend_lyrics
        self.feature_vector = []
        self.string_of_lyrics = lyrics

    def __str__(self) -> str:
        return self.label + ',' + str(self.song_name) + ',' + str(self.lyrics) + ',' + str(self.artist_id)

    def extract_unique_song_features(self, nouns: List, functionWords: List, wordAssociations: Dict, allEmotions: List, tfidf_transformer, vectorizer) -> None:
        """
        This method extracts a set of features:
            8 emotions (anger, fear, anticipation, trust, surprise, sadness, joy, disgust)
            2 polarities (negative, positive)
            length of longest word
            number of words in song
            repetition rate (unique_words/total_words)
            count of \n chars
            50 most frequent nouns in songs generally (love, time, way, day, ...) --> freq count
                (could also do binary for the above)
            50 most frequent function words (contains pronouns like eg. "I" and "you", determiners eg. "the" and "a" and prepositions eg. "in")
            5 punctuation symbols (',','.','!','?',''') --> same method as above
            1 count for song name length
            1 feature set to 1 for all songs (left this out)
            unigram tfidf score for a song
        """
        feat_vec = []
        emotion_feature_vector = feature_extraction.extract_emotions(self.lyrics, wordAssociations, allEmotions)
        longest_word_length_feature = feature_extraction.find_length_of_longest(self.lyrics)
        repetition_rate = feature_extraction.calculate_repetition_rate(self.lyrics)

        freq_nouns_count = feature_extraction.count_freq_nouns(self.lyrics, nouns)
        freq_func_count = feature_extraction.count_freq_nouns(self.lyrics, functionWords)
        freq_punct_count = feature_extraction.count_freq_nouns(self.lyrics, [",",".","!","?","'"])

        tfidf_score = feature_extraction.calculate_tfidf_score(tfidf_transformer, vectorizer, self.string_of_lyrics)
        #bi_tfidf_score = feature_extraction.calculate_tfidf_score(bi_tfidf_transformer, bi_vectorizer, self.string_of_lyrics)

        feat_vec += emotion_feature_vector
        feat_vec.append(longest_word_length_feature)
        feat_vec.append(len(self.lyrics))
        feat_vec.append(repetition_rate)
        feat_vec.append(self.lyrics.count("\n")) #count all \n chars, didn't need a method for that
        feat_vec += freq_nouns_count
        feat_vec += freq_func_count
        feat_vec += freq_punct_count
        feat_vec.append(len(self.song_name))
        #feat_vec.append(1)
        feat_vec.append(tfidf_score)
        #feat_vec.append(bi_tfidf_score)

        self.feature_vector = feat_vec

if __name__ == '__main__':
    s = Song('The Beatles', 1, 'Hills of green',
             "These are the \nlyrics of this song composed by a group of chaps. I think it's quite nice.")
    print(s.lyrics)

