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

    df = pd.DataFrame(data['data'][0:100])
    df['cleaned_summary'] = df.summary.apply(clean_text)
    df['cleaned_labels'] = df.categories.apply(clean_labels)
    print(df.head())

    sentences = df['cleaned_summary'].values
    labels = df['cleaned_labels'].to_list()

    # class_names = [label[0] for label in labels]  # Grab first category

    # Grab category that exists already
    class_names = []

    for label in labels:
        for category in label:
            if category in class_names:
                class_names.append(category)  # Append if already exists in list (reuse categories)
            else:
                class_names.append(label[0])  # Append first item
            break

    # Allocate training/test data
    training_sentences, test_sentences, training_categories, test_categories = train_test_split(
        sentences, class_names, test_size=0.25, random_state=1000)

    print('INPUT DATA')
    print(training_sentences[0])
    print(training_categories)

    print(len(training_sentences))
    print(len(training_categories))
    print()

    # Vectorize categories
    print('All training categories in sequential order are {}'.format(training_categories))
    print('Old training vocabulary has length {}'.format(len(training_categories)))
    print('New training vocabulary has length {}'.format(len(list(set(training_categories)))))
    print('The number of duplicates is {}'.format((len(training_categories) - len(list(set(training_categories))))))

    # Create lookup for them
    training_categories_lookup = {}

    for i, training_category in enumerate(list(set(training_categories))):
        training_categories_lookup[training_category] = i

    v_training_categories = np.array([training_categories_lookup[_] for _ in training_categories])

    # Vectorize sentences
    vectorizer = CountVectorizer(min_df=0, lowercase=False)
    vectorizer.fit(sentences)
    v_training_sentences = vectorizer.transform(training_sentences)

    print('TRAINING DATA')
    print(v_training_sentences.shape)
    print('Training Category Lookup: {}'.format(training_categories_lookup))
    print(v_training_categories, len(v_training_categories))
    print()

    input_dim = v_training_sentences.shape[1]

    print('INPUT DIM IS {}'.format(input_dim))

    model = Sequential()
    model.add(layers.Dense(128, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(len(v_training_categories), activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    history = model.fit(v_training_sentences, v_training_categories,
                        epochs=100,
                        verbose=False, batch_size=100)

    loss, accuracy = model.evaluate(v_training_sentences, v_training_categories, verbose=True)
    print("Training Accuracy: {:.4f}".format(accuracy))
