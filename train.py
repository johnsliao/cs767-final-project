import nltk
import json
import pandas as pd
import numpy as np
import string
import random
from keras.models import load_model

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

    df = pd.DataFrame(data['data'][0:100])
    df['cleaned_summary'] = df.summary.apply(clean_text)
    df['cleaned_labels'] = df.categories.apply(clean_labels)
    print(df.head())

    sentences = df['cleaned_summary'].values
    categories_lists = df['categories'].to_list()

    # class_names = [label[0] for label in labels]  # Grab first category

    # Grab category that exists already
    class_names = []
    banned_categories = ['Articles to be', 'lacking sources', 'births', 'all stub articles', 'all articles',
                         'Articles needing', 'articles with', 'Articles containing', 'All Wikipedia', 'Living people',
                         'Articles using', 'Commons category link', 'stub articles', 'All orphaned articles',
                         ]

    for category_list in categories_lists:
        cleaned_category_list = [x for x in category_list if
                                 not any(_.lower() in x.lower() for _ in banned_categories)]

        # Re-use dirty category list in none remain after cleaning
        if len(cleaned_category_list) == 0:
            cleaned_category_list = category_list

        for category in cleaned_category_list:
            if category in class_names:
                class_names.append(category)  # Append if already exists in list (reuse categories)
            else:
                class_names.append(cleaned_category_list[0])  # Append first item
            break

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

    # -- Build Model Start --

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

    print('INPUT DIM IS {}'.format(input_dim))

    model = Sequential()
    model.add(layers.Dense(128, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(len(v_categories), activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    history = model.fit(training_sentences, training_categories,
                        epochs=100,
                        verbose=False, batch_size=100)

    # Save model
    model.save('trained_model.h5')  # creates a HDF5 file 'my_model.h5'

    # -- Build Model End --

    # -- Test Model Start --

    # model = load_model('trained_model.h5')
    #
    # loss, accuracy = model.evaluate(training_sentences, training_categories, verbose=True)
    # print("Training Accuracy: {:.4f}".format(accuracy))
    #
    # test_loss, test_acc = model.evaluate(test_sentences, test_categories, verbose=2)
    # print('\nTest accuracy:', test_acc)
    #
    # predictions = model.predict(test_sentences)
    # print(sentences[75])
    # value = np.argmax(predictions[0])
    #
    # print(value)
    # print([k for k, v in categories_lookup.items() if v == value])

    # -- Test Model End --
