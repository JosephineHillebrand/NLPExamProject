"""
This script contains the code for classifying tweets as prochoice or prolife or neutral!.
The basic idea is to set up a number of pipelines for different types of classifiers.
These classifiers are then applied to different types of data in a 5-fold cross-validation, from which the average performance is extracted.
    1: Hashtags
    2: tweet text
    3: Hashtags in combination with tweet text
    4: Only the hashtags prochoice and prolife

The last part of the script contains the code for extracting
"""
#Import useful packages and functions
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import statistics

########### Initial operations #############
#Import data
df = pd.read_csv("C:/Users/hille/Desktop/NLP/Project/dfClean.csv")
df['Unnamed: 8'].value_counts()
df = df.dropna(subset=['hashtags'])
df = df.dropna(subset=['clean_text'])
#df = df.loc[df['Unnamed: 8'].isin([-1.0,1.0])]
df['lifeorchoice'] = np.nan
#remove row 1052 because something is wrong with it
df = df.drop(df.index[1052])


#Make a row in the dataframe only containing the hashtags prochoice and prolife, if these were used in the tweet
for i in range(0,len(df)):
    hashtags = []
    for hashtag in df.iloc[i]['hashtags'].split():
        if hashtag.lower() == '#prolife':
            hashtags.append("prolife")
        if hashtag.lower() == '#prochoice':
            hashtags.append("prochoice")
    df.iloc[i, -1] = ' '.join([str(elem) for elem in hashtags])


#Count the proportion of the prolife and prochoice tweets
df['Unnamed: 8'].value_counts()



#We create a new column with tokens

df['token_text'] = [
    [word for word in tweet.split()] for tweet in df['clean_text']]
print(df['token_text'])


#Find the average length of the cleaned tweets for prolife and prochoice tweets
tlprolife = []
tlprochoice = []
for i in range(0,len(df)):
    if df.iloc[i,8] == -1:
        tlprolife.append(len(df.iloc[i,18]))
    elif df.iloc[i,8] == 1:
        tlprochoice.append(len(df.iloc[i,18]))

statistics.mean(tlprolife) #Prolife
statistics.stdev(tlprolife)
statistics.mean(tlprochoice) #Prochoice
statistics.stdev(tlprochoice)

#Find the average word length for prolife and prochoice tweets
wlprolife = []
wlprochoice = []
for i in range(0,len(df)):
    if df.iloc[i,8] == -1:
        for w in df.iloc[i,18]:
            wlprolife.append(len(w))
    elif df.iloc[i,8] == 1:
        for w in df.iloc[i,18]:
            wlprochoice.append(len(w))

statistics.mean(wlprolife) #Prolife
statistics.stdev(wlprolife)
statistics.mean(wlprochoice) #Prochoice
statistics.stdev(wlprochoice)




########### Initial test of classification analysis in a simple train-test split #############

X_train, X_test, y_train, y_test = train_test_split(df['hashtags'], df['Unnamed: 8'])

#https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
#Above tutorial
#build a pipeline

######################## Build the four pipelines for classification purposes #######################
#THe first using a multinomial naive bayes classifier
MNB = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

#The second using a stochastic gradient decent linear classifier
SGD = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

#Logistic regression
logreg = Pipeline([('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(n_jobs =1, C=1e5, solver = 'lbfgs', max_iter = 10000)),
])

#Random forest
ranfor = Pipeline([('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier(n_estimators=100)),
])


#from sklearn import metrics
#print(metrics.classification_report(y_test, predicted))
#metrics.confusion_matrix(y_test, predicted)

text = 'hashtags'

#Creating our splits for a cross-validation
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0) #Returns the same results each time
#cv = ShuffleSplit(n_splits=5, test_size=0.2) #returns slightly different results each time

#The classification analysis has been inspired by:
#https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568

################# Cross validation analysis and results ######################

#Calculate all of the cross validation scores for all of the classifiers
#Only hashtags
scoresMNBh = cross_val_score(MNB, df['hashtags'], df['Unnamed: 8'], cv=cv)
scoresSGDh = cross_val_score(SGD, df['hashtags'], df['Unnamed: 8'], cv=cv)
scoreslogregh = cross_val_score(logreg, df['hashtags'], df['Unnamed: 8'], cv=cv)
scoresranforh = cross_val_score(ranfor, df['hashtags'], df['Unnamed: 8'], cv=cv)
#Only text
scoresMNBt = cross_val_score(MNB, df['clean_text'], df['Unnamed: 8'], cv=cv)
scoresSGDt = cross_val_score(SGD, df['clean_text'], df['Unnamed: 8'], cv=cv)
scoreslogregt = cross_val_score(logreg, df['clean_text'], df['Unnamed: 8'], cv=cv)
scoresranfort = cross_val_score(ranfor, df['clean_text'], df['Unnamed: 8'], cv=cv)
#Text and hashtags
scoresMNBth = cross_val_score(MNB, df['clean_text_hashtag_text'], df['Unnamed: 8'], cv=cv)
scoresSGDth = cross_val_score(SGD, df['clean_text_hashtag_text'], df['Unnamed: 8'], cv=cv)
scoreslogregth = cross_val_score(logreg, df['clean_text_hashtag_text'], df['Unnamed: 8'], cv=cv)
scoresranforth = cross_val_score(ranfor, df['clean_text_hashtag_text'], df['Unnamed: 8'], cv=cv)
#Prolife versus prochoice
scoresMNBlc = cross_val_score(MNB, df['lifeorchoice'], df['Unnamed: 8'], cv=cv)
scoresSGDlc = cross_val_score(SGD, df['lifeorchoice'], df['Unnamed: 8'], cv=cv)
scoreslogreglc = cross_val_score(logreg, df['lifeorchoice'], df['Unnamed: 8'], cv=cv)
scoresranforlc = cross_val_score(ranfor, df['lifeorchoice'], df['Unnamed: 8'], cv=cv)


#Print classification accuracy on hashtags, text, and both
print("Classification on hashtags")
print("Accuracy Multinomial Naive Bayes: \t {:.2f}".format(np.mean(scoresMNBh)))
print("Accuracy support vector machine: \t {:.2f}".format(np.mean(scoresSGDh)))
print("Accuracy logistic regression: \t\t {:.2f}".format(np.mean(scoreslogregh)))
print("Accuracy random forest classifier: \t {:.2f}".format(np.mean(scoresranforh)))

print("\n\nClassification on tweet text")
print("Accuracy Multinomial Naive Bayes: \t {:.2f}".format(np.mean(scoresMNBt)))
print("Accuracy support vector machine: \t {:.2f}".format(np.mean(scoresSGDt)))
print("Accuracy logistic regression: \t\t {:.2f}".format(np.mean(scoreslogregt)))
print("Accuracy random forest classifier: \t {:.2f}".format(np.mean(scoresranfort)))

print("\n\nClassification on hashtags and text")
print("Accuracy Multinomial Naive Bayes: \t {:.2f}".format(np.mean(scoresMNBth)))
print("Accuracy support vector machine: \t {:.2f}".format(np.mean(scoresSGDth)))
print("Accuracy logistic regression: \t\t {:.2f}".format(np.mean(scoreslogregth)))
print("Accuracy random forest classifier: \t {:.2f}".format(np.mean(scoresranforth)))

print("\n\nClassification on only prolife or prochoice hashtags")
print("Accuracy Multinomial Naive Bayes: \t {:.2f}".format(np.mean(scoresMNBlc)))
print("Accuracy support vector machine: \t {:.2f}".format(np.mean(scoresSGDlc)))
print("Accuracy logistic regression: \t\t {:.2f}".format(np.mean(scoreslogreglc)))
print("Accuracy random forest classifier: \t {:.2f}".format(np.mean(scoresranforlc)))


