import nltk
import json
import pandas as pd
import numpy as np
import string
from keras.utils import plot_model
import matplotlib.pyplot as plt

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
    with open('files/training_set.json', 'r') as fs:
        data = json.load(fs)

    df = pd.DataFrame(data['data'][0:10000])
    df['cleaned_summary'] = df.summary.apply(clean_text)
    df['cleaned_labels'] = df.categories.apply(clean_labels)
    print(df.head())

    sentences = df['cleaned_summary'].values
    categories_lists = df['categories'].to_list()

    class_names = [label[0] for label in categories_lists]  # Grab first category

    # Vectorize categories
    print('All categories in sequential order are {}'.format(class_names))
    print('Old category vocabulary has length {}'.format(len(class_names)))
    print('New category vocabulary has length {}'.format(len(list(set(class_names)))))
    print('The number of duplicates is {}'.format((len(class_names) - len(list(set(class_names))))))
    print('% duplicates is {}%'.format(100 * (len(class_names) - len(list(set(class_names)))) / len(class_names)))

    # Create lookup for them
    categories_lookup = {}

    for i, training_category in enumerate(list(set(class_names))):
        categories_lookup[training_category] = i

    v_categories = np.array([categories_lookup[_] for _ in class_names])

    # Vectorize Sentences
    vectorizer = CountVectorizer(min_df=0, lowercase=False)
    vectorizer.fit(sentences)
    v_sentences = vectorizer.transform(sentences)

    # Allocate training/test data
    training_sentences, test_sentences, training_categories, test_categories = train_test_split(
        v_sentences, v_categories, test_size=0.25, random_state=1000)

    print()
    print('INPUT DATA')
    print(training_sentences[0])
    print(training_categories)
    print()

    print('TRAINING DATA')
    print(training_sentences.shape)
    print('Category Lookup: {}'.format(v_categories))
    print(training_categories, len(training_categories))
    print()

    input_dim = training_sentences.shape[1]

    print('Input dimension is {}'.format(input_dim))

    model = Sequential()
    model.add(layers.Dense(128, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(len(v_categories), activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    history = model.fit(training_sentences, training_categories, epochs=10, verbose=1)
    print(history.history)
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    loss, accuracy = model.evaluate(training_sentences, training_categories, verbose=True)
    print("Training Accuracy: {:.4f}".format(accuracy))

    test_loss, test_acc = model.evaluate(test_sentences, test_categories, verbose=2)
    print('\nTest accuracy:', test_acc)
