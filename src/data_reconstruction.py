import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple

def filter_content(data:pd.DataFrame,k:int)->pd.DataFrame:
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
    top_k_artists_and_freq=Counter(dict_of_artist_song_freq).most_common(k)
    # reduced_dict = {artist:freq for artist, freq in dict_of_artist_song_freq.items() if freq > 150}
    #
    # reduced_data = data.loc[data['artist'].isin(reduced_dict.keys())]
    top_k_artists=[artist for artist,freq in top_k_artists_and_freq]
    reduced_data=data.loc[data['artist'].isin(top_k_artists)]
    return reduced_data

if __name__ == '__main__':
    content = pd.read_csv('../benchmark/songdata.csv', delimiter=',')
    print(filter_content(content))