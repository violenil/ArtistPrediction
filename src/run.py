
from DataSet import Dataset
import evaluation as eva

dataset = Dataset(file_path='../benchmark/songdata.csv')
resize_text = dataset.resize_data(list_of_song=dataset.list_of_songs)
list_artist_frequency = dataset.get_list_artist_to_index()
train_dt, validation_dt, test_dt = dataset.split_data(dt=resize_text)
unique_artists=list(dataset.artists_to_indx_dict.values()) # changed keys to values here to access numbers for each artist instead of name

batch_size = 10
list_of_train_batches = dataset.make_batches(data=train_dt, bch_sz=batch_size)

list_of_labels=[]
list_of_predicted_labels=[]
for batch in list_of_train_batches:
    labels = eva.extract_labels(batch)
    list_of_labels.extend(labels)
    predicted_labels = eva.artistPredictor(batch=batch, list_artist_frequency=list_artist_frequency)
    list_of_predicted_labels.extend(predicted_labels)
    #print(labels, '.....', predicted_labels)

evaluation = eva.evaluate_predictions(list_of_labels, list_of_predicted_labels, unique_artists) #this evaluation consists of dict of tp, fp, fn, tn

print(eva.micro_scores(evaluation))
print(eva.macro_scores(evaluation))
