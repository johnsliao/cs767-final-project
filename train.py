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

    df = pd.DataFrame(data['data'][0:100])
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

    # Vectorize labels
    pattern = "(?u)\\b[\\w-]+\\b"
    cv = CountVectorizer(lowercase=False, ngram_range=(1, 10), token_pattern=pattern)
    cv.fit(training_categories)
    label_vocab = cv.vocabulary_
    # print(label_vocab)
    # print(len(label_vocab))
    v_label_vocab = np.array([label_vocab[_] for _ in training_categories])
    print(len(v_label_vocab))
    print(v_label_vocab)

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

    history = model.fit(v_training_sentences, v_label_vocab,
                        epochs=100,
                        verbose=False, batch_size=10)

    # model.fit(training_sentences, training_categories, epochs=2)

    #
    # vectorizer = CountVectorizer(ngram_range=(2, 2))
    # vectorizer.fit(sentences_train)
    #
    # # Create BOW model vocabulary for training and testing set
    # X_train = vectorizer.transform(sentences_train)
    # X_test = vectorizer.transform(sentences_test)
    #
    # print(X_train)
    #
    # classifier = LogisticRegression()
    # classifier.fit(X_train, y_train)
    # score = classifier.score(X_test, y_test)
    # print(score)
