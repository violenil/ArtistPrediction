import pandas as pd
from Song import Song, embedding
from typing import List
import utilities as utt
from data_reconstruction import filter_content
import torch
import evaluation as eva
from tqdm import tqdm
from train_validation_test import run_epochs, no_of_top_artist
import numpy as np


### read file

def read_file(file_name: str) -> pd.DataFrame:
    """
    Function that reads a file and returns its content

    @param file_name:
    @return:
    """
    content = pd.read_csv(file_name, delimiter=',')
    return content


content = read_file(file_name='../benchmark/songdata.csv')

# artists_list=["Queen", "The Beatles", "Michael Jackson", "Eminem", "INXS"]
# content=content.loc[content['artist'].isin(artists_list)]
content = filter_content(data=content, k=no_of_top_artist)
content = content.sample(frac=1, random_state=7).reset_index(drop=True)  # shuffle data

no_of_artists = len(content['artist'].value_counts())
dict_artistnames_to_indx = {}
print('Read File')


def get_artist_to_idx(artist: str) -> int:
    """
    dictionary where the keys are artist_names and values are unique numbers.

    @param artist:
    @return:
    """
    if artist not in dict_artistnames_to_indx:
        dict_artistnames_to_indx[artist] = len(dict_artistnames_to_indx)
    return dict_artistnames_to_indx[artist]


content['artist_id'] = content['artist'].apply(
    get_artist_to_idx)  # a new column gets created in the csv file with unique artist_id.

### find feature vector length

### Create song instances and keep it in a list
list_of_song_instances = []
for i in tqdm(range(len(content)), desc='Creating Song Instances'):
    song_instance = Song(label=content.iloc[i][0], song_name=content.iloc[i][1], lyrics=content.iloc[i][3],
                         artist_id=content.iloc[i][4])
    list_of_song_instances.append(song_instance)

### Split Data
training_data_instances, validation_data_instances, test_data_instances = utt.split_data(dt=list_of_song_instances)
# print(len(training_data_instances), len(validation_data_instances))
batch_size = 15

### Make batches
training_data_instances_with_batches = utt.make_batches(data=training_data_instances, batch_size=batch_size)
validation_data_instances_with_batches = utt.make_batches(data=validation_data_instances, batch_size=batch_size)
test_data_instances_with_batches = utt.make_batches(data=test_data_instances, batch_size=batch_size)

# embedding_size = embedding.vector_size
embedding_size = 300
unique_artists = list(dict_artistnames_to_indx.values())

# def test_network(test_data_instances_with_batches: List) -> float:
#     list_of_actual_labels = []
#     list_of_predicted_labels = []
#     for batch in test_data_instances_with_batches:
#         list_of_labels = []
#         list_of_inputs = []
#         for song in batch:
#             label = song.artist_id
#             list_of_labels.append(label)
#             embedded_input = song.get_embedding()
#             list_of_inputs.append(embedded_input)
#         inputs = torch.FloatTensor(list_of_inputs)
#         labels = torch.FloatTensor(list_of_labels)
#         predicted_labels = classifier(embedded_input=inputs)
#         list_of_predicted_labels_per_batch = []
#         predicted_probabilities = classifier(embedded_input=inputs)
#
#         predicted_labels_per_batch = torch.argmax(predicted_probabilities, dim=1)
#         list_of_predicted_labels.extend(predicted_labels_per_batch.tolist())
#     return list_of_actual_labels, list_of_predicted_labels


# def extend_batches_to_list():

run_epochs(unique_artists, training_data_instances_with_batches, validation_data_instances_with_batches)
