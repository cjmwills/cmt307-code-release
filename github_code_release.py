import pandas as pd
import numpy as np
import requests

import sklearn
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_selection import chi2, SelectKBest

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)

np.random.seed(42)

# Read data

print("Reading data...")

def read_data(path_pos, path_neg):
    pos = pd.read_csv(path_pos, sep="\n", header=None, names=['review'])
    pos['positive']=1
    neg = pd.read_csv(path_neg, sep="\n", header=None, names=['review'])
    neg['positive']=0
    combined_df = pos.append(neg)
    combined_df = shuffle(combined_df, random_state=42)
    return(combined_df)


# Use the function to read the train, development and test sets.

train = read_data(path_pos="Data/IMDb/train/imdb_train_pos.txt",
                  path_neg="Data/IMDb/train/imdb_train_neg.txt")

test = read_data(path_pos="Data/IMDb/test/imdb_test_pos.txt",
                 path_neg="Data/IMDb/test/imdb_test_neg.txt")

# take set of stopwords from nltk
stopwords=set(nltk.corpus.stopwords.words('english'))
# manually add more punctuation
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")
stopwords.add("#")
stopwords.add("@")
stopwords.add(":")
stopwords.add("'s")
stopwords.add("’")
stopwords.add("...")
stopwords.add("n't")
stopwords.add("'re")
stopwords.add("'")
stopwords.add("-")
stopwords.add(";")
stopwords.add("/")
stopwords.add(">")
stopwords.add("<")
stopwords.add("br")
stopwords.add("(")
stopwords.add(")")
stopwords.add("''")
stopwords.add("&")

print("Preparing data...")

# Feature 1 - tf-idf

# Define custom transformers

class selectReview(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return(X['review'])

# Create transformation pipeline
feature_1_vocab = Pipeline([
    ('select_review', selectReview()),
    ('count', CountVectorizer(stop_words=stopwords, max_features=500)),
    ('tfidf', TfidfTransformer())
])


# Feature 2 - tf-idf (bi-grams)

feature_2_vocab = Pipeline([
    ('select_review', selectReview()),
    ('count', CountVectorizer(stop_words=stopwords, max_features=500, ngram_range=(2,2))),
    ('tfidf', TfidfTransformer())
])


# Feature 3 - Sentiment

vader = SentimentIntensityAnalyzer()

# Define custom transformers

class getSentiment(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        features_array=[]
        for index, row in X.iterrows():
            pos = vader.polarity_scores(row['review'])['pos']
            neu = vader.polarity_scores(row['review'])['neu']
            neg = vader.polarity_scores(row['review'])['neg']
            features_array.append([pos, neu, neg])
        return(np.asarray(features_array))

# Create transformation pipeline

feature_3_sentiment = Pipeline([
    ('get_sentiment', getSentiment())
])

# Combine all features

feature_engineering = FeatureUnion(transformer_list=[
    ("feature_1_vocab", feature_1_vocab),
    ("feature_2_vocab", feature_2_vocab),
    ("feature_3_sentiment", feature_3_sentiment)
])

# Run training data through the feature engineering pipeline to create transformed training set.

X_train = feature_engineering.fit_transform(train)
y_train = np.asarray(train['positive'])

# RBF

print("Building machine learning model...")

rbf_svm_clf = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("svm_clf", sklearn.svm.SVC(kernel="rbf"))
])

selector = SelectKBest(chi2, k=1000)
X_train_reduced = selector.fit_transform(X_train, y_train)

X_test = feature_engineering.transform(test)
X_test_reduced = selector.transform(X_test)

rbf_svm_clf.fit(X_train_reduced, y_train)

print("Making predictions...")

rbf_svm_clf_pred = rbf_svm_clf.predict(X_test_reduced)

print(rbf_svm_clf_pred)
