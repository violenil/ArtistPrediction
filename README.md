# Artist-prediction Project 🎵

Brief description about the project.

## How does it work? 🤔
The following steps gives a brief idea about the functional structure of the code.

- The dataset is read.
- Create instances for each song and keep it in a list.
- Create feature vector by considering words present in the lyrics of all songs.
- Create a class MCP where Perceptron instances per artist is created.
- The Perceptron calculates the scores based of feature vector and randomly initialized weight vector.
- The max score is then compared with the actual label
- Weights get updated in the Perceptron class.
- Evaluation is f-score.

## Requirements

###### Language
- python3

###### Libraries
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

###### Results
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
  - the micro_f_score or macro_f_score consists of list of scores, where each score corresponds to scores per epoch.

###### Plots
The plots are generated using matplotlib. no_of_epochs are plotted on the x-axis and f-scores are on the y-axis.


Apart from these while executing the run.py, the f-score per epoch is also visible in the terminal.

## Directory structure

- ```ArtistPrediction/benchmarks```
  - contains our dataset.

- ```ArtistPrediction/src/Song.py```
  - contains

- ```ArtistPrediction/src/Multi_class_Perceptron.py```
  - contains

- ```ArtistPrediction/src/Perceptron.py```
  - contains

- ```ArtistPrediction/src/evaluation.py```
  - contains

- ```ArtistPrediction/src/run.py```
  - contains

- ```ArtistPrediction/src/utilities.py```
  - contains

- ```ArtistPrediction/Plots```
  - contains

- ```ArtistPrediction/results```
  - here
