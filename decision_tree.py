import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix

# load dataset
dataset = pd.read_csv("train.csv")

# number of rows and columns
# print(dataset.shape)

# first five records
# print(dataset.head())

stopwords = set(stopwords.words('english'))

# cleaning data
def cleaning_data(dataset):
    cleaning = [char for char in dataset if char not in string.punctuation]
    cleaning = "".join(cleaning)
    return [word for word in cleaning.split() if word.lower() not in stopwords]

x_features = dataset['body']
# print(x_features)

y_labels = dataset['subreddit']
# print(y_labels)

vectorizer = CountVectorizer(analyzer=cleaning_data, max_features=5000).fit(x_features)


text_transform = vectorizer.transform(x_features)

X = text_transform.toarray()
# split data into training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y_labels,test_size=0.20)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
# print(y_pred)

# print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))