import pandas as pd
from Song import Song
import evaluation as eva
from Multi_class_Perceptron import MCP
from tqdm import tqdm
from data_reconstruction import filter_content, split_data
import plot_data as pl
import json
from datetime import datetime

print('Reading File')
content = pd.read_csv('../benchmark/songdata.csv', delimiter=',')
no_of_top_artist = 2
content = filter_content(data=content, k=no_of_top_artist)
content = content.sample(frac=1, random_state=7).reset_index(drop=True)  # shuffle data
dict_artistnames_to_indx = {}
print('Read File')
no_of_epochs = 1000


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
list_of_song_instances = []
for i in tqdm(range(len(content)), desc='Creating Song instances'):
    """
    Instances of each songs gets created. And the label, artist_id, song_name, lyrics are passed to the 
    respective class. Furthermore the instances are stored in a list. 
    """
    song_instance = Song(label=content.iloc[i][0], artist_id=content.iloc[i][4], song_name=content.iloc[i][1],
             lyrics=content.iloc[i][3])
    # print (s)
    list_of_song_instances.append(song_instance)
training_data, validation_data, test_data = split_data(dt=list_of_song_instances)

"""
Get feature vector for all songs
First create vocabulary of types in corpus of lyrics
"""
# list_of_feature_vectors = []
# vocab = []  # list of tokens
vocab = {}
s = set()
dict_of_word_count_in_all_songs = {}# a dictionary that contains unique words as keys
# and count of its presence in songs as values(if present repeated times in one song, the count of that word for that song is 1).
for song in tqdm(training_data, desc='Creating Vocab'):
    s1 = set(song.lyrics)
    s.update(s1)
    for i in s1:
        if i not in dict_of_word_count_in_all_songs:
            dict_of_word_count_in_all_songs[i] = 0
        dict_of_word_count_in_all_songs[i] = dict_of_word_count_in_all_songs[i] + 1
marginal_length = len(list_of_song_instances) * 0.8 # need this length to find the words that is present in 80% of the songs.
for word in s:
    if dict_of_word_count_in_all_songs[word] < marginal_length:
        vocab[word] = len(vocab)
print(len(vocab))

for song in \
        tqdm(training_data, desc='Creating Feature_vector for training'):
    song.bow_feature_extraction(vocab)
    # list_of_feature_vectors.append(song.feature_vector)
    # print(song.feature_vector)
for song in tqdm(validation_data, desc='Creating Feature_vector for validation'):
    song.bow_feature_extraction(vocab)

for song in tqdm(test_data, desc='Creating Feature_vector for testing'):
    song.bow_feature_extraction(vocab)

"""
Running  Multi_class_Perceptron
"""
total_classes = list(dict_artistnames_to_indx.keys())
m = MCP(classes=total_classes, weight_vec_length=len(vocab))

unique_artists = list(dict_artistnames_to_indx.values())
list_of_evaluation_micro_scores = []
list_of_evaluation_macro_scores = []

for epoch in range(no_of_epochs):
    list_of_predicted_labels = []
    list_of_actual_labels = []

    for song in tqdm(training_data, desc='Training'):
        feature_vec = [0] * len(vocab)
        for idx in song.feature_vector:
            feature_vec[idx] = 1
        label = song.label
        dict_of_scores = m.find_all_scores(feature_vec)
        predicted_perceptron, max_score = m.find_max(dict_of_scores=dict_of_scores)
        actual_perceptron = label
        # print(f'predicted={predicted_perceptron}, actual={actual_perceptron}')
        m.update_weight(actual_perceptron=actual_perceptron, predicted_perceptron=predicted_perceptron,
                        feature_vec=feature_vec)
    for song in tqdm(validation_data, desc='Validating'):
        feature_vec=[0]*len(vocab)
        for idx in song.feature_vector:
            feature_vec[idx]=1
        label=song.label
        dict_of_scores=m.find_all_scores(feature_vec)
        predicted_perceptron, max_score = m.find_max(dict_of_scores=dict_of_scores)
        list_of_actual_labels.append(song.artist_id)
        list_of_predicted_labels.append(dict_artistnames_to_indx[predicted_perceptron])

    evaluation = eva.evaluate_predictions(list_of_actual_labels, list_of_predicted_labels,
                                          unique_artists)
    micro_scores_dict = eva.micro_scores(evaluation)
    macro_scores_dict = eva.macro_scores(evaluation)

    print(
        f'epoch= {epoch + 1}, micro_f_score= {micro_scores_dict["microF1"]}, macro_f_score= {macro_scores_dict["macroF1"]}')
    list_of_evaluation_micro_scores.append(micro_scores_dict['microF1'])
    list_of_evaluation_macro_scores.append(macro_scores_dict['macroF1'])
pl.plot_data(list_of_scores=list_of_evaluation_macro_scores, yaxis_label='macro_scores',
             file_name=str(no_of_top_artist) + '_artists_macro_' + str(no_of_epochs))
pl.plot_data(list_of_scores=list_of_evaluation_micro_scores, yaxis_label='micro_scores',
             file_name=str(no_of_top_artist) + '_artists_micro_' + str(no_of_epochs))
with open('../results/' + str(no_of_top_artist) + '_artists_' + str(no_of_epochs) + '.json',
          "w") as results_file:
    now = datetime.now()
    date_time_as_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt = {
        'ran_on': date_time_as_string,
        'no_of_artists': no_of_top_artist,
        'no_of_epochs': no_of_epochs,
        'macro_f_score': list_of_evaluation_macro_scores,
        'micro_f_score': list_of_evaluation_micro_scores
    }
    json.dump(dt, results_file)

list_of_actual_labels=[]
list_of_predicted_labels=[]
for song in tqdm(test_data, desc='Testing'):
    feature_vec=[0]*len(vocab)
    for idx in song.feature_vector:
        feature_vec[idx]=1
    label=song.label
    dict_of_scores=m.find_all_scores(feature_vec)
    predicted_perceptron, max_score = m.find_max(dict_of_scores=dict_of_scores)
    list_of_actual_labels.append(song.artist_id)
    list_of_predicted_labels.append(dict_artistnames_to_indx[predicted_perceptron])

evaluation = eva.evaluate_predictions(list_of_actual_labels, list_of_predicted_labels,
                                      unique_artists)
micro_scores_dict = eva.micro_scores(evaluation)
macro_scores_dict = eva.macro_scores(evaluation)
print(f'Testing f-scores Micro_f_score = {micro_scores_dict["microF1"]} Macro_f_score =  {macro_scores_dict["macroF1"]}')

"""
Below is where the evaluation of the results happens.
We create a list of predicted labels based on the frequencies of labels within the data.
The 'evaluation' variable consists of comparing Gold Standard to Predicted labels for each class (each artist)
and then we calculate micro and macro precision, recall and F scores. These are what is returned by
micro_scores and macro_scores.
"""
list_of_artist_frequency = list(content['artist_id'])
list_of_labels = list(content['artist_id'])
list_of_predicted_labels = eva.artistPredictor(list_artist_frequency=list_of_artist_frequency)
unique_artists = list(dict_artistnames_to_indx.values())

evaluation = eva.evaluate_predictions(list_of_labels, list_of_predicted_labels,
                                      unique_artists)  # this evaluation consists of dict of tp, fp, fn, tn

micro_scores_dict = eva.micro_scores(evaluation)
macro_scores_dict = eva.macro_scores(evaluation)

# print(micro_scores_dict)
# print(macro_scores_dict)
