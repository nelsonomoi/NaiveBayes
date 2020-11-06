import pandas as pd
import string
from numpy import mean,std
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

# check feature names
tokens = vectorizer.get_feature_names()

text_transform = vectorizer.transform(training_data['body'])

X = text_transform.toarray()

# convert training_data into  a dataframe
# dd = pd.DataFrame(training_data=text_transform.toarray(),columns=tokens)
# print(dd)

# cross validation
kf = KFold(n_splits=10,random_state=1,shuffle=True)

# for train, test in kf.split(X):
#     print('train: %s, test: %s' %(train,test))

# split training data
x_train, x_test, y_train, y_test = train_test_split(X, training_data.subreddit, test_size=0.2, random_state=50)
# print(x_train.shape)

# create model fit
model = BernoulliNB().fit(x_train,y_train)

# model predict
prediction = model.predict(x_test)

# print(prediction)
# test accuracy score
acc_score = accuracy_score(prediction,y_test)
print(acc_score)
print(classification_report(y_test, prediction))

# cross  validation score
scores = cross_val_score(model,X,training_data.subreddit,scoring="accuracy",cv=kf,n_jobs=1)
print("Accuracy: %.3f (%.3f)"%(mean(scores),std(scores)))


# evaluation
# load test.csv
test_data = pd.read_csv('test.csv')
# user_data = ['I am a training_data scientist with two years of experience. I figured some people might get some from my experience so here it goes.   I graduated from NYU with a degree in finance and applied to about 5 jobs to start with. A lot of them were actually Finance jobs at Fidelity  Morgan Stanley  JP Morgan  Mellon and then the company I work with.   The jobs available at Fidelity and Mellon were not really entry level  so it didnâ€™t surprise me that I did not get those jobs.   I did get a job at Morgan Stanley  JP Morgan and my current company. I rejected the former two and accepted the latter because they had an office close to my parents house  which was a benefit for someone just starting out and because it was a general training_data scientist position rather than more of a financial analyst position.  At that point I had a background in Stats  Calc and R through coursework in high school and uni and I had a calendar year of experience doing research in a financial sector and several months as a Psychology RA. I understood all of the common types of models (Linear Regression  KNN  Time Series etc.) very well and I had applied them.   During my first year at work I mostly supported research projects. My local office didnâ€™t have another training_data scientist (they still donâ€™t) and I had to build the capability from the ground up. This involved a lot of research as well as meeting people where they were. Many of the people I was working with were still hand entering training_data into a spreadsheet and then painstakingly hand jamming graphs. I supported these employees by teaching them how to build better spreadsheets for analytics  automating processes in spreadsheets by teaching myself VBA and helping the business/financial etc. analysts understand what kind of things would prevent fully a fully automated process. There were some business cases that involved some simple modeling  but nothing too crazy.   During this time I also took a graduate course in Python that covered all of the basic statistical techniques (LR  KNN Time Series etc.) using my education benefit. I was bored  so the professor sent me more advanced materials. I learned more about Neural Networks  SMOTE and other techniques.  The first 1/2 of my second year I worked mostly with visualizations. I had helped my colleagues get their spreadsheets into a more usable format and automated some of the training_data pulls (not all of the training_data systems are hosted on a server that can be attached to a viz tool~ modernization is still in progress). Before the end of my first year there had been some â€œdashboardsâ€ but they mostly consisted of slapping graphs onto a tableau dashboard all over creation. Some departments did not have the budget/ leadership backing for Tableau so we stuck to automated spreadsheets  but for the departments that did have the budget and the support I created dashboards that were well thought out  had purpose  flowed  included various features to make them extremely easy to use and I made that the standard for']

test_unseen = vectorizer.transform(test_data['body'])
# test_1_unseen = vectorizer.transform(user_data)

unseen_data_to_array = test_unseen.toarray()

predict_unseen = model.predict(unseen_data_to_array)

print(predict_unseen)
# end evaluation

