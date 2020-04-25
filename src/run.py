
from DataSet import Dataset
import evaluation as eva

dataset = Dataset(file_path='../benchmark/songdata.csv')
# print(dataset)
resize_text = dataset.resize_data(list_of_song=dataset.list_of_songs)
list_artist_frequency = dataset.get_list_artist_to_index()
train_dt, validation_dt, test_dt = dataset.split_data(dt=resize_text)
# print(train_dt, '\n', validation_dt, '\n', test_dt)
batch_size = 10
list_of_train_batches = dataset.make_batches(data=train_dt, bch_sz=batch_size)
# print(list_of_train_batches)

for batch in list_of_train_batches:
    correct = eva.extract_labels(batch)
    predictions = eva.artistPredictor(batch=batch, list_artist_frequency=list_artist_frequency)
    print(correct, '.....', predictions)

    evaluat = eva.evaluation(predictions, correct)
    p = evaluat[0] / (evaluat[0] + evaluat[1])
    r = evaluat[0] / (evaluat[0] + evaluat[2])
    if p==0 or r==0:
        fscore=0
    else:
        fscore = 2 * ((p * r) / (p + r))

    print("Precision: " + str(p))
    print("Recall: " + str(r))
    print("F-score: " + str(fscore))