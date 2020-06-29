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
import tf_idf_feature_extraction as tf_idf


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

# artists_list = ["Queen", "The Beatles", "Michael Jackson", "Eminem", "INXS"]
# content = content.loc[content['artist'].isin(artists_list)]
content = filter_content(data=content, k=no_of_top_artist)
content = content.sample(frac=1, random_state=7).reset_index(drop=True)  # shuffle data

dict_artistnames_to_indx = {}


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

#### SPLIT DATA
training_content_len = int(len(content) * 0.9)
validate_content_len = int(len(content) * 0.05)
training_content = content[:training_content_len]
validate_content = content[training_content_len:training_content_len + validate_content_len]
testing_content = content[training_content_len + validate_content_len:]

no_of_artists = len(content['artist'].value_counts())
print('Read File')

### TF-IDF column
training_content = tf_idf.get_tf_idf_values(training_content)


### Create song instances and keep it in a list
def create_song_instances(content: pd.DataFrame):
    list_of_song_instances = []
    # for i in tqdm(range(len(content)), desc='Creating Song Instances'):
    for i, song in content.iterrows():
        tf_idf = 0.0
        if 'tf_idf_score' in song:
            tf_idf = song['tf_idf_score']
        song_instance = Song(label=song['artist'], song_name=song['song'], lyrics=song['text'],
                             artist_id=song['artist_id'], tf_idf_score=tf_idf)
        list_of_song_instances.append(song_instance)
    return list_of_song_instances


### call create instances using the split data separately
training_data_instances = create_song_instances(content=training_content)
validation_data_instances = create_song_instances(content=validate_content)
test_data_instances = create_song_instances(content=testing_content)
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

run_epochs(unique_artists, training_data_instances_with_batches, validation_data_instances_with_batches, test_data_instances_with_batches)
