# Artist-prediction Project 🎵

This projects makes an attempt at classifying song instances by artist using only their lyrics and song names. New song instances should then be predicted as being produced by a particular artist.

The most challenging part of this project is the large number classes in the dataset. There are a total of 645 artists, some of which do not have a representative sample of songs (ie. less than 50), making it quite difficult to distinguish between different classes.

Previous work that relates to this project includes authorship attribution, where characterizations of documents are defined that capture the writing style of authors. This project attempts to do a similar thing in that it focuses on the aspects of song lyrics (and song names) that are unique to each artist. Other work that relates specifically to music and lyrics is song genre classification, where the goal is to distinguish between a given number of genres and predict which genre a song falls into.

The approach of this project was to first build a Perceptron classifier, a simple baseline, which considers only a bag-of-words binary feature vector in its classification decision. At a later point, more complex feature vectors may be extracted to increase the efficiency of classification.

## How does it work? 🤔
The following steps gives a brief idea about the functional structure of the code.

- The dataset is read.
- Instances for each song are created and kept in a list.
- Feature vectors are created by considering words present in the lyrics of all songs (bag-of-words approach)
- An instance of the Multi-Class Perceptron (MCP) is created, which controls the creation of one Perceptron for each class.
- Each Perceptron calculates the scores based on a given feature vector and randomly initialized weight vector.
- The max score is then compared with the actual label.
- Weights get updated in the Perceptron class.
- The Evaluation returns micro and macro f-scores.

## Requirements

#### Language
- python3

#### Libraries
- pandas
- re
- matplotlib
- tqdm


## How to run? 🏃

In order to run the code, the file named run.py should be executed. Following are the steps of instructions, how to execute run.py.

1. Open a terminal and change directory to _src_. ``` cd ArtistPrediction/src ```

2. Execute the  command.  ``` python3 run.py ```

## Interpreting the results

After the execution of run.py gets completed, several files are created in the directory results and Plots.

#### Results
The evaluated f-scores are kept in a json file and looks like ```
{
  "ran_on": "18/05/2020 15:18:18",
  "no_of_artists": 2,
  "no_of_epochs": 100,
  "macro_f_score": [
    0.5102112720296128, 0.5129003851385256, ....]}```

   __Interpreting the json files__
  - The file consists of four keys. the date the file got created, no_of_artists, no_of_epochs, either macro_f_score or micro_f_score.
  - The 'no_of_artists' and 'no_of_epochs' keys consists of integers as values.
  - the 'micro_f_score' or 'macro_f_score' consists of list of scores, where each score corresponds to scores per epoch.

#### Plots
The plots are generated using matplotlib. no_of_epochs are plotted on the x-axis and f-scores are on the y-axis.


Apart from these while executing the run.py, the f-score per epoch is also visible in the terminal.

## Directory structure

- ```ArtistPrediction/benchmarks```
  - contains our dataset.

- ```ArtistPrediction/src```
  - contains our scource code.

- ```ArtistPrediction/Plots```
  - contains the generated plots

- ```ArtistPrediction/results```
  - contains json files with results of every epoch.
