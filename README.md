# Code Release

This GitHub repo contains the code and data needed to predict the sentiment of a moview review as positive or negative. It builds a Radial Basis Function Support Vector Machine (RBF SVM) using 1003 features;

- 500 unigrams.
- 500 bigrams.
- 3 sentiment scores.

The 1000 most relevant features, calculated using the Chi-squared test, are selected to build the model.

## Requirements

To run this machine learning model you will need the following installed;

1. Git
2. Python (with `sklearn`, `pandas`, `numpy` and `nltk` libraries.)

## Modifying Test Data

By default the model will use the dataset in the `Data/IMDb/test/` folder. To run the model on different data, simply replace this file with the data you want the model to predict.

## Model Parameters

[The model parameters](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) can be changed by editing line 143 of `model.py`.

## Running the Model

To run the model complete the following steps;

1. Clone the repository to your local machine.
2. Navigate to the repository in the terminal.
3. Type `python model.py` and hit enter.
