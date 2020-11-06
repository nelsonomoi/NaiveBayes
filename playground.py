import pandas as pd
import string
from numpy import mean, std
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

stopwords = set(stopwords.words('english'))

# load csv training_data
training_data = pd.read_csv('train.csv', sep=",")


# unique word computers
# print(training_data['subreddit'].value_counts())

# cleaning data
def cleaning_data(dataset):
    cleaning = [char for char in dataset if char not in string.punctuation]
    cleaning = "".join(cleaning)
    return [word for word in cleaning.split() if word.lower() not in stopwords]


vectorizer = CountVectorizer(analyzer=cleaning_data, max_features=5000).fit(training_data['body'])

text_transform = vectorizer.transform(training_data['body'])

X = text_transform.toarray()

print(X)