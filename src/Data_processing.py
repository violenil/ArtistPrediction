import pandas as pd
from typing import List, Dict, Tuple

PAD_STR = '__UNK__'


def read_file(file_path: str) -> pd.DataFrame:
    content = pd.read_csv(file_path, delimiter=',')
    return content


def get_data(content: pd.DataFrame) -> List:
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
        tok_song_name = get_tokenized_data(song_text=song_name)

        lyrics = content.iloc[i][3]
        # the next line is written for removing all the '\n'
        new_lyrics = "".join(lyrics.splitlines())
        tok_new_lyrics = get_tokenized_data(new_lyrics)
        list_song_data.append(artist_name)
        list_song_data.append(tok_song_name)
        list_song_data.append(tok_new_lyrics)
        list_of_songs.append(list_song_data)
    return list_of_songs


def get_tokenized_data(song_text: str) -> List:
    """
    called from get_data each time while we retrieve data from 'content' and return the tokens of the data.
    """
    tokens = song_text.split(' ')
    # tokens = get_tokens(text=song_text, chars=[' ', ',', "'"])
    return tokens


# def get_tokens(text: str, chars: List = [' ']) -> List:
#     tokens = []
#
#     return tokens

def resize_data(list_of_song: List) -> List:
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
    avg_song_name_length = find_average_length(list_length_song_name)
    avg_lyrics_length = find_average_length(list_length_lyrics)
    for song in list_of_song:
        song[1] = set_length(song[1], avg_song_name_length)
        song[2] = set_length(song[2], avg_lyrics_length)
    return list_of_song


def find_average_length(list_of_length: List) -> int:
    """
    Find average of all the song data, song names or lyrics.
    """
    sum_of_length = 0
    for each_len in list_of_length:
        sum_of_length = sum_of_length + each_len
    average_length = sum_of_length // len(list_of_length)
    return average_length


def set_length(song_data: List, fixed_length: int) -> List:
    """
    Fix the length of each song data, song names or lyrics.
    """
    nl = song_data[:fixed_length]
    return nl + [PAD_STR] * (fixed_length - len(song_data))


def find_unique_artist(resized_text: List) -> Tuple[List,Dict, Dict]:
    """
    We find unique number of artists in our data set and assign each of them to separate numbers.

    return: a dict with unique numbers as keys and unique artists as their values.
    """
    list_of_artist = []
    for song in resized_text:
        list_of_artist.append(song[0])
    dict_artistnames_to_indx = {}
    dict_indx_to_atistnames = {}
    for artist in list_of_artist:
        if artist not in dict_artistnames_to_indx:
            val = len(dict_artistnames_to_indx) + 1
            dict_artistnames_to_indx[artist] = val
            dict_indx_to_atistnames[val] = artist
    return list_of_artist, dict_artistnames_to_indx, dict_indx_to_atistnames


def change_labels_to_numbers(resized_text:List, dict_artistnames_to_indx:Dict)->List:
    """
    We change our artists_names in data, to their respective values in dict_artistnames_to_indx.

    :param resized_text:
    :param artists_dict:
    :return:
    """
    for song in resized_text:
        song[0]=dict_artistnames_to_indx[song[0]]
    return resized_text

def list_artist_to_index(list_of_artist:list,dict_artist_to_indx:dict)->list:
    """
    we create a list_of_artist where we just replace the artists_names with
    there respective values in the dict_artist_to_indx. We use this list later
    when we randomly predict the artists for pre evaluation. 
    :param list_of_artist:
    :param dict_artist_to_indx:
    :return:
    """
    list_artist_freq=[]
    for artist in list_of_artist:
        list_artist_freq.append(dict_artist_to_indx[artist])
    return list_artist_freq

if __name__ == "__main__":
    content = read_file(file_path='../benchmark/fewsongs.csv')
    # content = read_file('../benchmark/songdata.csv')
    # print (content)
    text = get_data(content=content)
    # print(text)
    resize_text = resize_data(list_of_song=text)
    # print(resize_text)
    list_artist, artists_to_indx_dict, index_to_artist_dict = find_unique_artist(resized_text=resize_text)
    # print(artists_dict)
    data = change_labels_to_numbers(resized_text=resize_text, dict_artistnames_to_indx=artists_to_indx_dict)
    print(data)
    list_artist_frequency=list_artist_to_index(list_of_artist=list_artist,dict_artist_to_indx=artists_to_indx_dict)
