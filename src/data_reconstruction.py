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
    top_k_artists=[artist for artist,freq in top_k_artists_and_freq]
    reduced_data=data.loc[data['artist'].isin(top_k_artists)]
    return reduced_data

def split_data(dt:List) ->Tuple[List,List,List]:
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

def make_batches(data:List, batch_size:int) -> List:
    per_batch = []
    list_with_batch = []
    for e in data:
        per_batch.append(e)
        if len(per_batch) == batch_size:
            list_with_batch.append(per_batch)
            per_batch = []
    return list_with_batch



if __name__ == '__main__':
    content = pd.read_csv('../benchmark/songdata.csv', delimiter=',')
    print(filter_content(content))