import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from google.cloud import storage

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download("stopwords")

BUCKET='my-training-artifacts'
DATA_LOCATION='gs://my-training-artifacts/WELFake_Dataset.csv'

# loading and preparing data
news_df = pd.read_csv(DATA_LOCATION)
news_df.fillna('', inplace=True)
news_df['content'] = [x + ' ' + y for x,y in zip(news_df.title, news_df.text)]

# cleaning and pre-processing data
def clean_and_prepare_content(text):
    text = re.sub('[^a-zA-Z]',' ', text)
    text = text.lower()
    text_words = text.split()
    imp_text_words = [word for word in text_words if not word in stopwords.words('english')]
    stemmed_words = [porter_stemmer.stem(word) for word in imp_text_words]
    processed_text = ' '.join(stemmed_words)
    return processed_text

porter_stemmer = PorterStemmer()
news_df['processed_content'] = news_df.content.apply(lambda content: clean_and_prepare_content(content))

# separating data and labels
X = news_df.processed_content.values
y = news_df.label.values
print(X.shape, y.shape)

# converting data into numerical format
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

# splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=42)

# define and fit RF model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# saving model file to GCS
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET)
blob = bucket.blob("rf_model.pkl")
with blob.open(mode="wb") as file:
    pickle.dump(rf_model, file)

# predict on test set
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)

# print classification report
print(
    classification_report(
        y_test,
        y_pred,
        target_names=['Real', 'Fake'],
    )
)

# print confusion matrix
print(confusion_matrix(y_test, y_pred,))
