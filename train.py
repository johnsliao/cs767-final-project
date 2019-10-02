import nltk
import json
import pandas as pd
import string
from tensorflow import keras
import unicodedata
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers
from tensorflow import feature_column
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer as tf
from keras.models import Sequential
from keras import layers
import numpy as np


def remove_punct(text):
    return "".join([char for char in text if char not in string.punctuation])


def remove_stopwords(tokenized_list):
    stopword = nltk.corpus.stopwords.words('english')
    return [word for word in tokenized_list if word not in stopword]


def stemming(tokenized_text):
    ps = nltk.PorterStemmer()
    return [ps.stem(word) for word in tokenized_text]


def clean_text(text):
    # print(text)
    text = remove_punct(text)
    text = nltk.word_tokenize(text)
    tokens = remove_stopwords(text)
    text = stemming(tokens)
    text = ' '.join(text)
    return text


def clean_labels(labels):
    new_labels = []
    for label in labels:
        new_labels.append(remove_punct(label))
    return new_labels


if __name__ == '__main__':
    with open('files/test_set.json', 'r') as fs:
        data = json.load(fs)

    df = pd.DataFrame(data['data'][0:1000])
    df['cleaned_summary'] = df.summary.apply(clean_text)
    df['cleaned_labels'] = df.categories.apply(clean_labels)
    print(df.head())

    sentences = df['cleaned_summary'].values
    labels = df['cleaned_labels'].to_list()

    single_labels = [label[0] for label in labels]  # Grab first category

    # Allocate training/test data
    training_sentences, test_sentences, training_categories, test_categories = train_test_split(
        sentences, single_labels, test_size=0.25, random_state=1000)

    # print(len(training_set_sentences))
    # print(training_set_sentences)

    # Vectorize categories
    cv = CountVectorizer(lowercase=False, ngram_range=(1, 10), token_pattern="(?u)\\b[\\w-]+\\b")
    cv.fit(training_categories)
    label_vocab = cv.vocabulary_
    v_train_category_vocab = np.array([label_vocab[_] for _ in training_categories])

    cv = CountVectorizer(lowercase=False, ngram_range=(1, 10), token_pattern="(?u)\\b[\\w-]+\\b")
    cv.fit(test_categories)
    label_vocab = cv.vocabulary_
    v_test_category_vocab = np.array([label_vocab[_] for _ in test_categories])

    # Vectorize sentences
    vectorizer = CountVectorizer(min_df=0, lowercase=False)
    vectorizer.fit(sentences)
    v_training_sentences = vectorizer.transform(training_sentences)

    print(v_training_sentences.shape)

    input_dim = v_training_sentences.shape[1]

    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    history = model.fit(v_training_sentences, v_train_category_vocab,
                        epochs=100,
                        verbose=False, batch_size=10)

    loss, accuracy = model.evaluate(v_training_sentences, v_train_category_vocab, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
