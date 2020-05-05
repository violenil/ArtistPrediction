import pandas as pd
import random
from typing import List, Dict, Tuple
from nltk.tokenize import word_tokenize

PAD_STR = '__UNK__'
random.seed=1

class Dataset:
    def __init__(self, file_path: str) -> None:
        text = self.read_file(file_path=file_path)
        songs = self.get_data(content=text)
        random.shuffle(songs)
        self.list_of_songs = songs
        self.list_artist, self.artists_to_indx_dict, self.index_to_artist_dict = self.find_unique_artist()
        self.change_labels_to_numbers()

    def __str__(self):
        '''
        Return any string
        '''
        l = self.list_of_songs[:2]
        return str(l)

    def __len__(self):
        return len(self.list_of_songs)

    def read_file(self, file_path: str) -> pd.DataFrame:
        content = pd.read_csv(file_path, delimiter=',')
        return content

    def get_data(self, content: pd.DataFrame) -> List:
        """
        :retrieve required data from each row one by one
        :remove the '\n' of new line which are present in lyrics.
        :tokenize text while retrieving the data (we do that in a separate function by calling each time
        when we retrieve data)

        :return: a list of song with its tokenized data. format: [[artist,[song_name tokens],[lyrics_tokens]],[],[],....[]]
        """
        list_of_songs = []
        for i in range(len(content)):
            list_song_data = []
            artist_name = content.iloc[i][0]
            song_name = content.iloc[i][1]
            tok_song_name = self.get_tokenized_data(song_text=song_name)

            lyrics = content.iloc[i][3]
            # the next line is written for removing all the '\n'
            #new_lyrics = "".join(lyrics.splitlines())
            tok_new_lyrics = self.get_tokenized_data(lyrics)
            list_song_data.append(artist_name)
            list_song_data.append(tok_song_name)
            list_song_data.append(tok_new_lyrics)
            list_of_songs.append(list_song_data)
        return list_of_songs

    def get_tokenized_data(self, song_text: str) -> List:
        """
        called from get_data each time while we retrieve data from 'content' and return the tokens of the data.
        """
        tokens = word_tokenize(song_text)

        #tokens = song_text.split(' ')
        # tokens = get_tokens(text=song_text, chars=[' ', ',', "'"])
        return tokens

    # def get_tokens(text: str, chars: List = [' ']) -> List:
    #     tokens = []
    #
    #     return tokens

    def resize_data(self, list_of_song: List) -> List:
        """
        Here we resize the data ,i.e. we want to have a fixed length of all the song names and the fixed length of
        all the song lyrics.
        hence we find average of song_name_length and average of lyrics_length and set the length of all the respective
        song data by truncating the lists or appending a new word to the list as per requirement.

        return: new resized list of songs.
        """
        list_length_song_name = []
        list_length_lyrics = []
        for song in list_of_song:
            list_length_song_name.append(len(song[1]))
            list_length_lyrics.append(len(song[2]))
        avg_song_name_length = self.find_average_length(list_length_song_name)
        avg_lyrics_length = self.find_average_length(list_length_lyrics)
        for song in list_of_song:
            song[1] = self.set_length(song[1], avg_song_name_length)
            song[2] = self.set_length(song[2], avg_lyrics_length)
        return list_of_song

    def find_average_length(self, list_of_length: List) -> int:
        """
        Find average of all the song data, song names or lyrics.
        """
        sum_of_length = 0
        for each_len in list_of_length:
            sum_of_length = sum_of_length + each_len
        average_length = sum_of_length // len(list_of_length)
        return average_length

    def set_length(self, song_data: List, fixed_length: int) -> List:
        """
        Fix the length of each song data, song names or lyrics.
        """
        nl = song_data[:fixed_length]
        return nl + [PAD_STR] * (fixed_length - len(song_data))

    def find_unique_artist(self) -> Tuple[List, Dict, Dict]:
        """
        We find unique number of artists in our data set and assign each of them to separate numbers.

        return: a dict with unique numbers as keys and unique artists as their values.
        """
        list_of_artist = []
        for song in self.list_of_songs:
            list_of_artist.append(song[0])
        dict_artistnames_to_indx = {}
        dict_indx_to_atistnames = {}
        for artist in list_of_artist:
            if artist not in dict_artistnames_to_indx:
                val = len(dict_artistnames_to_indx) + 1
                dict_artistnames_to_indx[artist] = val
                dict_indx_to_atistnames[val] = artist
        return list_of_artist, dict_artistnames_to_indx, dict_indx_to_atistnames

    def change_labels_to_numbers(self):
        """
        We change our artists_names in data, to their respective values in dict_artistnames_to_indx.

        :param list_of_songs:
        :param artists_dict:
        :return:
        """
        for song in self.list_of_songs:
            song[0] = self.artists_to_indx_dict[song[0]]

    def get_list_artist_to_index(self) -> list:
        """
        we create a list_of_artist where we just replace the artists_names with
        there respective values in the dict_artist_to_indx. We use this list later
        when we randomly predict the artists for pre evaluation.
        :param list_of_artist:
        :param dict_artist_to_indx:
        :return:
        """
        list_artist_freq = []
        for artist in self.list_artist:
            list_artist_freq.append(self.artists_to_indx_dict[artist])
        return list_artist_freq

    def split_data(self, dt: List) -> Tuple[List, List, List]:
        """
        split data into train and validation and test data. We make a 80,10,10 split.
        :param dt: The list of our retrieved and processed data
        :return: total training, validation and test data.
        """
        total_train_data_len = int(len(dt) * 0.8)
        total_validation_data_len = int(len(dt) * 0.1)
        total_test_data_len = len(dt) - (total_train_data_len + total_validation_data_len)
        total_train_data = dt[:total_train_data_len]
        total_validation_data = dt[total_train_data_len:total_train_data_len + total_validation_data_len]
        total_test_data = dt[total_validation_data_len + total_train_data_len:]
        return total_train_data, total_validation_data, total_test_data

    def make_batches(self, data: List, bch_sz: int) -> List:
        """
        We make batches of the training data and validation data according to the batch size. It simply is another list inside
        the list of songs where each batch size number of songs are keep in another list. We discard the excess data in the end
        that does not form a batch.
        :param data:
        :param bch_sz:
        :return: list of songs with batches. [songs[batch1[song data 1],[2],[3]],[[],[],[]],......[batch n[],[],[]]]
        """
        a = []
        list_with_batches = []
        for e in data:
            a.append(e)
            if len(a) == bch_sz:
                list_with_batches.append(a)
                a = []
        return list_with_batches


if __name__ == "__main__":
    dataset = Dataset(file_path='../benchmark/fewsongs.csv')
    # print(dataset)
    resize_text = dataset.resize_data(list_of_song=dataset.list_of_songs)
    list_artist_frequency = dataset.get_list_artist_to_index()
    train_dt, validation_dt, test_dt = dataset.split_data(dt=resize_text)
    # print(train_dt, '\n', validation_dt, '\n', test_dt)
    unique_artist=set(dataset.list_artist)
    batch_size = 2
    list_of_train_batches = dataset.make_batches(data=train_dt, bch_sz=batch_size)
    # print(list_of_train_batches)

