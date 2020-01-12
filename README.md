# Code Release

## Requirements

To run this machine learning model you will need the following installed;

1. Git
2. Python (with `sklearn`, `pandas`, `numpy` and `nltk` libraries.)

## Modifying Test Data

By default the model will use the dataset in the `Data/IMDb/test/` folder. To run the model on different data, simply replace this file with the data you want the model to predict.

## Model Parameters

The model runs a Radial Basis Function Support Vector Machine (RBF SVM). [The model parameters](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) can be changed by editing `model.py`.

## Running the Model

To run the model complete the following steps;

1. Clone the repository to your local machine.
2. Navigate to the repository in the terminal.
3. Type `python model.py` and hit enter.
