# Artist-prediction Project

This repository contains the project and related details for our NLP TeamLab course, where we build a Perceptron classifier from scratch and predict artists based on a large corpus of song lyrics.

## Data Processing

- Here we read the .csv file and extract the features of the songs(artist name, son name and lyrics) that we want to work with. Each songs are kept in a list, and their data are further seperately organized in a list of list format. Further more we tokenize the retrieved data.

- After that we resize the data by fixing the length of the lists, inside the list of songs. For example, if we have a list of songs with 2 songs, the song data are also in seperate lists inside it as -->

```[[['what','is','your','name'],['i','am','supriti']],[['who','are','you'],['its','girl']]]```
resizing should give

```[[['what','is','your'],['i','am']],[['who','are','you'],['its','girl']]]```
we do this by taking an average length of ```song[0][0]``` , ```song[1][0]``` and ```song[0][1]```, ```song[1][1]``` and so on if more features are present.

- Then we find the unique artists from all the songs we retrieved and map them to a unique intezer. We then replace the artist names in our original list with their assigned number. Our data is now processed and organized to work with.
