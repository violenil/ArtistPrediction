import pandas as pd
from typing import List, Dict, Tuple

def filter_content(data:pd.DataFrame)->pd.DataFrame:
    """
    select only few songs based on some criteria.
    @param data:
    @return:
    """
    dict_of_artist_song_freq = {}
    all_artists = data['artist']
    for n, artist in all_artists.items():
        if artist in dict_of_artist_song_freq.keys():
            dict_of_artist_song_freq[artist] += 1
        else:
            dict_of_artist_song_freq[artist] = 1
    
    reduced_dict = {artist:freq for artist, freq in dict_of_artist_song_freq.items() if freq > 150}
    
    return reduced_dict

if __name__ == '__main__':
    content = pd.read_csv('../benchmark/songdata.csv', delimiter=',')
    filter_content(content)
    print(reduced_dict)