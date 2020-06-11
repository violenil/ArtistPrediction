import evaluation as eva
from tqdm import tqdm
import torch
import numpy as np
from Classifier import Classifier
from typing import List
from evealuation2 import evaluation_of_prediction
from plot_data import plot_trainig_validation_loss

"""
TRAINING
"""

device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')

classifier = Classifier(embedding_size=300, no_of_labels=5)
classifier.to(device)
lr = 5.0  # initial learning rate
optimizer = torch.optim.Adam(classifier.parameters(),
                             lr=lr)  # gradient descend how much ,towards which side (calculate weight update value)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 1, gamma=0.95)


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
            embedded_input = song.get_embedding()
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
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 0.5)
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
                embedded_input = song.get_embedding()
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


def run_epochs(embedding_size, unique_artists, training_data_instances_with_batches,
               validation_data_instances_with_batches):
    no_of_epoch = 500
    list_mean_loss_of_training = []
    list_of_mean_loss_of_validation = []

    for epoch in range(no_of_epoch):
        mean_loss_of_training = train_network(training_data_instances_with_batches)
        labels, predicted_labels, mean_loss_of_validation = validate_network(validation_data_instances_with_batches)
        evaluation,accuracy = eva.evaluate_predictions(labels, predicted_labels,
                                              unique_artists)
        micro_scores_dict = eva.micro_scores(evaluation)
        macro_scores_dict = eva.macro_scores(evaluation)

        print(
            f'epoch= {epoch + 1}, micro_f_score= {micro_scores_dict["microF1"]}, macro_f_score= {macro_scores_dict["macroF1"]},ac={evaluation["ACC"]}, lr={scheduler.get_last_lr()[0]}')
        list_mean_loss_of_training.append(mean_loss_of_training)
        list_of_mean_loss_of_validation.append(mean_loss_of_validation)
        print(mean_loss_of_training, '\n', mean_loss_of_validation)
        scheduler.step()
    # print(list_mean_loss_of_training, '\n', list_of_mean_loss_of_validation)
    plot_trainig_validation_loss(list_mean_loss_of_training, list_of_mean_loss_of_validation)
    # labels, predicted_labels = test_network(test_data_instances_with_batches)
    # evaluation = eva.evaluate_predictions(labels, predicted_labels,
    #                                       unique_artists)
    # micro_scores_dict = eva.micro_scores(evaluation)
    # macro_scores_dict = eva.macro_scores(evaluation)
    # print('micro_f_score= ', micro_scores_dict["microF1"], 'macro_f_score= ', macro_scores_dict["macroF1"])
