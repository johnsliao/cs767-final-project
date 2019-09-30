import nltk
import json
import pandas as pd
import string
import unicodedata
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers
from tensorflow import feature_column
from sklearn.linear_model import LogisticRegression


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


with open('files/test_set.json', 'r') as fs:
    data = json.load(fs)

df = pd.DataFrame(data['data'][0:100])
df['cleaned_summary'] = df.summary.apply(clean_text)
print(df.head())

# Vectorize sentences
sentences = df['cleaned_summary'].values

vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences)

# Vectorize labels
labels = df['categories'].to_list()
single_labels = []

# Only using first category...
for label in labels:
    single_labels.append(label[0])

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, single_labels, test_size=0.25,
                                                                    random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print(score)
