from DataSet import Dataset
import evaluation as eva

dataset = Dataset(file_path='../benchmark/songdata.csv')
print(len(dataset.list_of_songs))
resize_text = dataset.resize_data(list_of_song=dataset.list_of_songs)
list_artist_frequency = dataset.get_list_artist_to_index()
train_dt, validation_dt, test_dt = dataset.split_data(dt=resize_text)
unique_artists = list(dataset.index_to_artist_dict.keys())

batch_size = 10
list_of_train_batches = dataset.make_batches(data=train_dt, bch_sz=batch_size)

list_of_labels = []
list_of_predicted_labels = []
for batch in list_of_train_batches:
    labels = eva.extract_labels(batch)
    list_of_labels.extend(labels)
    predicted_labels = eva.artistPredictor(batch=batch, list_artist_frequency=list_artist_frequency)
    list_of_predicted_labels.extend(predicted_labels)
    # print(labels, '.....', predicted_labels)

evaluation = eva.evaluate_predictions(list_of_labels, list_of_predicted_labels, unique_artists) #this evaluation consists of dict of tp, fp, fn, tn

#micro_scores_dict = eva.micro_scores(evaluation)
#macro_scores_dict = eva.macro_scores(evaluation)

#print(micro_scores_dict)
#print(macro_scores_dict)


print(dataset.list_of_songs[0][2])
