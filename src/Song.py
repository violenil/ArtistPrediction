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
    
    def extract_emotions(self, wordAssociations: Dict, allEmotions: List) -> List:
        """
        first find all emotions that come up for the words in your lyrics (may want to exclude function words here using pos)
        then count occurrence of all 8 emotions and neg and positive
        return these as a list of counts of emotions in the given order --> list of length 10
        """
        #reducedTokenTagDict = {word:tag for word, tag in self.tagged_lyrics.items() if tag in ['NN', 'NNP', 'NNS', 'PRP', 'RB', 'JJ', 'JJS', 'VB', 'VBD', 'VBG']}
        #reducedLyrics = list(reducedTokenTagDict.keys())
        #I wanted to only process those words that have a particular tag, but the dict doesnt keep duplicates of words so this is not possible
        #TO DO
        
        reducedLyrics = self.lyrics
        observed_emotions = [] # instances of all associated emotions

        for word in reducedLyrics:
            if word in wordAssociations.keys():
                observed_emotions += wordAssociations[word]

        emotion_feature_vector = []
        for e in allEmotions:
            emotion_feature_vector.append(observed_emotions.count(e))
        assert len(emotion_feature_vector) == 10
        return(emotion_feature_vector)
  
    def find_length_of_longest(self):
        """
        search list for longest word and return its length
        """
        #lyrics_list = list(self.tagged_lyrics.keys())
        lyrics_list = self.lyrics
        return(len(max(lyrics_list, key=len)))
    
    def calculate_repetition_rate(self):
        """
        repetition_rate would be the number of unique words in the lyrics divided by the total
        number of words in the lyrics
        """
        #lyrics_list = list(self.tagged_lyrics.keys())
        lyrics_list = self.lyrics
        return(int((len(set(lyrics_list))/len(lyrics_list))*10))

    def count_freq_nouns(self, popNouns: List) -> List:
        """
        takes a list of popular nouns in songs (10 of them) and counts how frequently they occur in 
        the lyrics, returning the list of counts
        """
        popCount = []
        for w in popNouns:
            popCount.append(self.lyrics.count(w))
        return(popCount)

    def extract_unique_song_features(self, wordAssociations: Dict, allEmotions: List) -> None:
        """
        This method extracts a set of features:
            8 emotions (anger, fear, anticipation, trust, surprise, sadness, joy, disgust)
            2 polarities (negative, positive)
            length of longest word in song
            repetition rate (unique_words/total_words)
            count of \n chars
            10 most frequent nouns in songs generally (love, time, way, day, baby, heart, life, night, gonna, man) --> freq count
                (could also do binary for the above)
            counts for 5 punctuation symbols (',','.','!','?',''') --> I added this to the above for now (same method)
        """
        feat_vec = []
        emotion_feature_vector = self.extract_emotions(wordAssociations, allEmotions)
        longest_word_length_feature = self.find_length_of_longest()
        repetition_rate = self.calculate_repetition_rate()

        popular_nouns = ["love", "time", "way", "day", "baby", "heart", "life", "night", "gonna", "man"] 
        freq_nouns_count = self.count_freq_nouns(popular_nouns)
        freq_punct_count = self.count_freq_nouns([",",".","!","?","'"]) 
        
        feat_vec += emotion_feature_vector
        feat_vec.append(longest_word_length_feature)
        feat_vec.append(repetition_rate)
        feat_vec.append(self.lyrics.count("\n")) #count all \n chars, didn't need a method for that
        feat_vec += freq_nouns_count
        feat_vec += freq_punct_count

        self.feature_vector = feat_vec


if __name__ == '__main__':
    s = Song('The Beatles', 1, 'Hills of green',
             "These are the \nlyrics of this song composed by a group of chaps. I think it's quite nice.")
    print(s.lyrics)
