import nltk
import json
import string
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint

def remove_punct(text):
    return "".join([char for char in text if char not in string.punctuation])


def remove_stopwords(tokenized_list):
    stopword = nltk.corpus.stopwords.words('english')
    return [word for word in tokenized_list if word not in stopword]


def stemming(tokenized_text):
    ps = nltk.PorterStemmer()
    return [ps.stem(word) for word in tokenized_text]


def clean_text(text):
    print(text)
    text = remove_punct(text)
    text = nltk.word_tokenize(text)
    tokens = remove_stopwords(text)
    text = stemming(tokens)
    text = [' '.join(text)]
    return text


sample = {'name': 'Michael Hardt',
          'summary': 'Michael Hardt (born 1960) is an American political philosopher and literary theorist. Hardt is best known for his book Empire, which was co-written with Antonio Negri. It has been praised by Slavoj Žižek as the "Communist Manifesto of the 21st Century".Hardt and Negri suggest that several forces which they see as dominating contemporary life, such as class oppression, globalization and the commodification of services (or production of affects), have the potential to spark social change of unprecedented dimensions. A sequel, Multitude: War and Democracy in the Age of Empire was published in August 2004. It outlines an idea first propounded in Empire, which is that of the multitude as possible locus of a democratic movement of global proportions. The third and final part of the trilogy, Commonwealth, was published in 2009.',
          'categories': ['1960 births', '20th-century American philosophers', '20th-century American writers',
                         '21st-century American non-fiction writers', '21st-century American philosophers',
                         'All BLP articles lacking sources', 'All articles with unsourced statements',
                         'American Marxists', 'American communists', 'American literary critics',
                         'American political philosophers', 'Articles with hCards',
                         'Articles with unsourced statements from October 2015', 'Autonomism',
                         'BLP articles lacking sources from October 2013', 'Continental philosophers',
                         'Duke University faculty', 'European Graduate School faculty', 'Libertarian socialists',
                         'Living people', 'Marxist theorists', 'People from Bethesda, Maryland',
                         'People from Potomac, Maryland', 'Swarthmore College alumni',
                         'University of Washington alumni', 'Wikipedia articles with BNF identifiers',
                         'Wikipedia articles with GND identifiers', 'Wikipedia articles with ISNI identifiers',
                         'Wikipedia articles with LCCN identifiers', 'Wikipedia articles with NDL identifiers',
                         'Wikipedia articles with NKC identifiers', 'Wikipedia articles with NTA identifiers',
                         'Wikipedia articles with SELIBR identifiers', 'Wikipedia articles with SUDOC identifiers',
                         'Wikipedia articles with VIAF identifiers',
                         'Wikipedia articles with WorldCat-VIAF identifiers']}

with open('files/test_set.json', 'r') as fs:
    data = json.load(fs)

# Pre-processing data
# https://towardsdatascience.com/natural-language-processing-nlp-for-machine-learning-d44498845d5b
ngram_vect = CountVectorizer(ngram_range=(2, 2), analyzer=clean_text)
x_counts = ngram_vect.fit_transform([sample['summary'], ])
print(x_counts.shape)
print(ngram_vect.get_feature_names())

#
# tagged = nltk.pos_tag(tokens)
# print(tagged)
