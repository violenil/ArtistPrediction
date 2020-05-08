import pandas as pd
from Song import Song
import evaluation as eva

content = pd.read_csv('../benchmark/fewsongs.csv', delimiter=',')
content = content.sample(frac=1).reset_index(drop=True)  # shuffle data
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
list_of_song_instances = []
for i in range(len(content)):
    """
    Instances of each songs gets created. And the label, artist_id, song_name, lyrics are passed to the 
    respective class. Furthermore the instances are stored in a list. 
    """
    s = Song(label=content.iloc[i][0], artist_id=content.iloc[i][4], song_name=content.iloc[i][1],
             lyrics=content.iloc[i][3])
    # print (s)
    list_of_song_instances.append(s)

list_of_artist_frequency = list(content['artist_id'])
list_of_labels = list(content['artist_id'])
list_of_predicted_labels = eva.artistPredictor(list_artist_frequency=list_of_artist_frequency)

unique_artists = list(dict_artistnames_to_indx.values())

evaluation = eva.evaluate_predictions(list_of_labels, list_of_predicted_labels,
                                      unique_artists)  # this evaluation consists of dict of tp, fp, fn, tn

micro_scores_dict = eva.micro_scores(evaluation)
macro_scores_dict = eva.macro_scores(evaluation)

print(micro_scores_dict)
print(macro_scores_dict)
