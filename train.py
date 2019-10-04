import nltk
import json
import pandas as pd
import numpy as np
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers


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

    df = pd.DataFrame(data['data'][0:10])
    df['cleaned_summary'] = df.summary.apply(clean_text)
    df['cleaned_labels'] = df.categories.apply(clean_labels)
    print(df.head())

    sentences = df['cleaned_summary'].values
    labels = df['cleaned_labels'].to_list()

    class_names = [label[0] for label in labels]  # Grab first category

    # Allocate training/test data
    training_sentences, test_sentences, training_categories, test_categories = train_test_split(
        sentences, class_names, test_size=0.25, random_state=1000)

    print('INPUT DATA')
    print(training_sentences)
    print(training_categories)

    print(len(training_sentences))
    print(len(training_categories))
    print()

    # Vectorize categories
    v_training_categories = np.array([x for x, y in enumerate(training_categories)])  # Use index of class name

    # Vectorize sentences
    vectorizer = CountVectorizer(min_df=0, lowercase=False)
    vectorizer.fit(sentences)
    v_training_sentences = vectorizer.transform(training_sentences)

    print('TRAINING DATA')
    print(v_training_sentences.shape)
    print(v_training_categories)
    print()

    input_dim = v_training_sentences.shape[1]

    print('INPUT DIM IS {}'.format(input_dim))

    model = Sequential()
    model.add(layers.Dense(128, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    history = model.fit(v_training_sentences, v_training_categories,
                        epochs=100,
                        verbose=False, batch_size=100)

    loss, accuracy = model.evaluate(v_training_sentences, v_training_categories, verbose=True)
    print("Training Accuracy: {:.4f}".format(accuracy))
