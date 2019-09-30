import nltk
import json
import pandas as pd
import string
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
vectorizer = CountVectorizer(min_df=0, lowercase=False)
sentences = df['cleaned_summary'].values
vectorizer.fit(sentences)

# Vectorize labels
labels = df['categories'].to_list()
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.25,
                                                                    random_state=1000)


