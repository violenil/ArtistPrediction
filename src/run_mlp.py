import pandas as pd
from Song import Song
from typing import List
import data_reconstruction as dr


### read file
def read_file(file_name: str) -> pd.DataFrame:
    return content

content = read_file(file_name='fewsongs.csv')

### find feature vector length
def find_feature_vec_length(lyrics: pd.Series) -> int:
    length=0
    return length

find_feature_vec_length(lyrics=content['text'])

### Create song instances and keep it in a list
list_of_song_instances=[]
for i in range(len(content)):
    song_instance=Song(label=content.iloc[i][0], song_name=content.iloc[i][1], lyrics=content.iloc[i][3])
    list_of_song_instances.append(song_instance)

### Split Data
training_data_instances, validation_data_instances, test_data_instances = dr.split_data(dt=list_of_song_instances)
batch_size=2

### Make batches
training_data_instances_with_batches=dr.make_batches(data=training_data_instances, batch_size=batch_size)
validation_data_instances_with_batches=dr.make_batches(data=validation_data_instances, batch_size=batch_size)
test_data_instances_with_batches=dr.make_batches(data=test_data_instances, batch_size=batch_size)

### training
def train_network(training_data_instances_with_batches:List) -> float:
    for batch in training_data_instances_with_batches:
        label=extract_label_per_batch(batch)
        embeding=Song.get_embedding(lyrics)
        predicted_labels=call_my_classifier()
        #calculate loss

### validation
def validate_network(validation_data_instances_with_batches) -> float:
    for batch in validation_data_instances_with_batches:
        label=extract_label_per_batch(batch)
        embedding=Song.get_embedding(lyrics)
        predicted_labels=call_my_classifier()
    #evaluation

### testing
def test_network(test_data_instances_with_batches: List) -> float:
    for batch in test_data_instances_with_batches:
        label=extract_lable_per_batch(batch)
        embeding=Song.get_embedding(lyrics)
        predicted_labels=call_my_classifier()
    #evaluation

for i in epoch:
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
