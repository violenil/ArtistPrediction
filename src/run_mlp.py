import pandas as pd
from Song import Song, embedding
from typing import List
import utilities as utt
from Classifier import Classifier
import torch


### read file
def read_file(file_name: str) -> pd.DataFrame:
    """
    Function that reads a file and returns its content

    @param file_name:
    @return:
    """
    content = pd.read_csv(file_name, delimiter=',')
    return content


content = read_file(file_name='../benchmark/fewsongs.csv')
print(content)
no_of_artists = len(content['artist'].value_counts())
print(no_of_artists)
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


### find feature vector length
def resize_lyrics(lyrics: str) -> str:
    """
    @param lyrics:
    @return:
    """
    tokenized_lyrics=lyrics.split(' ')
    length = 75
    resized_lyrics=' '.join(tokenized_lyrics[:length])
    return resized_lyrics


content['text']=content['text'].apply(resize_lyrics)
print(content['text'])
### Create song instances and keep it in a list
list_of_song_instances = []
for i in range(len(content)):
    song_instance = Song(label=content.iloc[i][0], song_name=content.iloc[i][1], lyrics=content.iloc[i][3],
                         artist_id=content.iloc[i][4])
    list_of_song_instances.append(song_instance)

### Split Data
training_data_instances, validation_data_instances, test_data_instances = utt.split_data(dt=list_of_song_instances)
batch_size = 2

### Make batches
training_data_instances_with_batches = utt.make_batches(data=training_data_instances, batch_size=batch_size)
validation_data_instances_with_batches = utt.make_batches(data=validation_data_instances, batch_size=batch_size)
test_data_instances_with_batches = utt.make_batches(data=test_data_instances, batch_size=batch_size)

### training
embedding_size = embedding.vector_size

classifier = Classifier(embedding_size=embedding_size, no_of_labels=no_of_artists)


def train_network(training_data_instances_with_batches: List) -> float:
    for batch in training_data_instances_with_batches:
        list_of_labels = []
        list_of_inputs = []
        for song in batch:
            label = song.artist_id
            list_of_labels.append(label)
            embedded_input = song.get_embedding()
            list_of_inputs.append(embedded_input)
        # inputs = torch.tensor(list_of_inputs)
        # labels = torch.tensor(list_of_labels)
        predicted_labels = classifier(embedded_input=list_of_inputs)
        # calculate loss


### validation
def validate_network(validation_data_instances_with_batches) -> float:
    for batch in validation_data_instances_with_batches:
        list_of_labels = []
        list_of_inputs = []
        for song in batch:
            label = song.artist_id
            list_of_labels.append(label)
            embedded_input = song.get_embedding()
            list_of_inputs.append(embedded_input)
        inputs = torch.FloatTensor(list_of_inputs)
        labels = torch.FloatTensor(list_of_labels)
        predicted_labels = classifier(embedded_input=inputs)
    # evaluation


### testing
def test_network(test_data_instances_with_batches: List) -> float:
    for batch in test_data_instances_with_batches:
        list_of_labels = []
        list_of_inputs = []
        for song in batch:
            label = song.artist_id
            list_of_labels.append(label)
            embedded_input = song.get_embedding()
            list_of_inputs.append(embedded_input)
        inputs = torch.FloatTensor(list_of_inputs)
        labels = torch.FloatTensor(list_of_labels)
        predicted_labels = classifier(embedded_input=inputs)
    # evaluation


epoch = 10
for i in range(epoch):
    train_network(training_data_instances_with_batches)
    validate_network(validation_data_instances_with_batches)
test_network(test_data_instances_with_batches)
"""
read file
find feature vec length
create instances of songs
split data
make batches
Song.py()
    padding and truncating
    embedding
training
    epoch - input
        batch
            get embedding
            get labels
            call classifier
            loss calc

validation
    same as training + evaluation
"""
