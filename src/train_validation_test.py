import evaluation as eva
from tqdm import tqdm
import torch
import numpy as np
from Classifier import Classifier
from Classifier_with_choosen_feature_vector import Classifier_manual_features
from typing import List
from plot_data import plot_trainig_validation_loss
import plot_data as pl
import json
from datetime import datetime
from Classifier_with_cnn import Classifier_using_cnn

"""
TRAINING
"""
no_of_top_artist = 2
no_of_epochs = 40
model_name = ['RNN', 'manual_features', 'CNN'][2]  # also change song.py

device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')

if model_name == 'RNN':
    classifier = Classifier(embedding_size=300, no_of_labels=no_of_top_artist)  # to access the classifier that uses RNN
elif model_name == 'manual_features':
    classifier = Classifier_manual_features(embedding_size=121,
                                            no_of_labels=no_of_top_artist)  # To access the classifier that uses manual_features
elif model_name == 'CNN':
    classifier = Classifier_using_cnn(embedding_size=300, no_of_labels=no_of_top_artist)

classifier.to(device)
# lr = 5.0  # initial learning rate
optimizer = torch.optim.Adam(
    classifier.parameters())  # gradient descend how much ,towards which side (calculate weight update value)


# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)


def train_network(training_data_instances_with_batches: List) -> float:
    classifier.train()  # must write to update weights each time
    list_of_loss = []
    for batch in tqdm(training_data_instances_with_batches, desc='Training'):
        optimizer.zero_grad()  # initial weight update set to zero.
        list_of_labels_per_batch = []
        list_of_inputs = []
        for song in batch:
            label = song.artist_id
            list_of_labels_per_batch.append(label)
            embedded_input = song.get_feature_vector(model_name)
            list_of_inputs.append(embedded_input)
        inputs = torch.FloatTensor(list_of_inputs)
        inputs = inputs.to(device)
        predicted_probabilities = classifier(embedded_input=inputs)

        # calculate loss
        labels = torch.tensor(list_of_labels_per_batch).long().to(device)
        loss = classifier.loss_calc(predicted_probabilities,
                                    labels)  # have to clear how here we can add parameter although no parameter stated while nn.NLLoss?
        list_of_loss.append(loss.item())
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(classifier.parameters(), 0.5)
        optimizer.step()
    mean_loss = np.mean(list_of_loss)
    return mean_loss


### validation
def validate_network(validation_data_instances_with_batches: List) -> float:
    classifier.eval()
    list_of_actual_labels = []
    list_of_predicted_labels = []
    list_of_loss = []
    with torch.no_grad():
        for batch in tqdm(validation_data_instances_with_batches, desc='Validating'):
            list_of_labels_per_batch = []
            list_of_inputs = []
            for song in batch:
                label = song.artist_id
                list_of_labels_per_batch.append(label)
                embedded_input = song.get_feature_vector(model_name)
                list_of_inputs.append(embedded_input)
            inputs = torch.FloatTensor(list_of_inputs)
            inputs = inputs.to(device)
            # labels = torch.tensor(list_of_labels)
            predicted_probabilities = classifier(embedded_input=inputs)
            labels = torch.tensor(list_of_labels_per_batch).long().to(device)

            loss = classifier.loss_calc(predicted_probabilities, labels)
            list_of_loss.append(loss.item())
            predicted_labels_per_batch = torch.argmax(predicted_probabilities, dim=1)
            list_of_predicted_labels.extend(predicted_labels_per_batch.tolist())
            list_of_actual_labels.extend(list_of_labels_per_batch)
        mean_loss = np.mean(list_of_loss)

    return list_of_actual_labels, list_of_predicted_labels, mean_loss


### TESTING
def test_network(test_data_instances_with_batches: List) -> float:
    classifier.eval()
    list_of_actual_labels = []
    list_of_predicted_labels = []
    with torch.no_grad():  # no backpropagation
        for batch in tqdm(test_data_instances_with_batches, desc='Testing'):
            list_of_labels_per_batch = []
            list_of_inputs = []
            for song in batch:
                label = song.artist_id
                list_of_labels_per_batch.append(label)
                embedded_input = song.get_feature_vector(model_name)
                list_of_inputs.append(embedded_input)
            inputs = torch.FloatTensor(list_of_inputs)
            inputs = inputs.to(device) # if present sent to GPU
            # labels = torch.FloatTensor(list_of_labels)
            predicted_probabilities = classifier(embedded_input=inputs)
            #labels = torch.tensor(list_of_labels_per_batch).long().to(device)
            predicted_labels_per_batch = torch.argmax(predicted_probabilities, dim=1)
            list_of_predicted_labels.extend(predicted_labels_per_batch.tolist())
            list_of_actual_labels.extend(list_of_labels_per_batch)
            # list_of_predicted_labels_per_batch = []
            # predicted_probabilities = classifier(embedded_input=inputs)
            #
            # predicted_labels_per_batch = torch.argmax(predicted_probabilities, dim=1)
            # list_of_predicted_labels.extend(predicted_labels_per_batch.tolist())
    return list_of_actual_labels, list_of_predicted_labels


def run_epochs(unique_artists, training_data_instances_with_batches,
               validation_data_instances_with_batches, test_data_instances_with_batches):
    list_of_evaluation_micro_scores = []
    list_of_evaluation_macro_scores = []
    list_mean_loss_of_training = []
    list_of_mean_loss_of_validation = []

    for epoch in range(no_of_epochs):
        mean_loss_of_training = train_network(training_data_instances_with_batches)
        labels, predicted_labels, mean_loss_of_validation = validate_network(validation_data_instances_with_batches)
        evaluation = eva.evaluate_predictions(labels, predicted_labels,
                                              unique_artists)
        micro_scores_dict = eva.micro_scores(evaluation)
        macro_scores_dict = eva.macro_scores(evaluation)

        print(
            f'epoch= {epoch + 1}, micro_f_score= {micro_scores_dict["microF1"]}, macro_f_score= {macro_scores_dict["macroF1"]}')
        list_mean_loss_of_training.append(mean_loss_of_training)
        list_of_mean_loss_of_validation.append(mean_loss_of_validation)
        print(mean_loss_of_training, '\n', mean_loss_of_validation)
        # scheduler.step()
        list_of_evaluation_micro_scores.append(micro_scores_dict['microF1'])
        list_of_evaluation_macro_scores.append(macro_scores_dict['macroF1'])
    # print(list_mean_loss_of_training, '\n', list_of_mean_loss_of_validation)

    pl.plot_data(list_of_scores=list_of_evaluation_macro_scores, yaxis_label='macro_scores',
                 file_name=classifier.name + '_' + str(no_of_top_artist) + '_artists_macro_' + str(no_of_epochs))
    pl.plot_data(list_of_scores=list_of_evaluation_micro_scores, yaxis_label='micro_scores',
                 file_name=classifier.name + '_' + str(no_of_top_artist) + '_artists_micro_' + str(no_of_epochs))
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
    plot_trainig_validation_loss(list_mean_loss_of_training, list_of_mean_loss_of_validation)
    list_of_actual_labels, list_of_predicted_labels=test_network(test_data_instances_with_batches)
    evaluation = eva.evaluate_predictions(list_of_actual_labels, list_of_predicted_labels,
                                          unique_artists)
    micro_scores_dict = eva.micro_scores(evaluation)
    macro_scores_dict = eva.macro_scores(evaluation)
    print(
        f'Testing f-scores Micro_f_score = {micro_scores_dict["microF1"]} Macro_f_score =  {macro_scores_dict["macroF1"]}')
