#Import useful stuff
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np

#Import data
df = pd.read_csv("C:/Users/hille/Desktop/NLP/Project/dfClean.csv")
df['Unnamed: 8'].value_counts()
df = df.dropna(subset=['hashtags'])
df = df.dropna(subset=['clean_text'])
df = df.loc[df['Unnamed: 8'].isin([-1.0,1.0])]
df['lifeorchoice'] = np.nan


#Make a row in the dataframe only nothing whether there is a prochoice, prolife or both hashtags
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
"""df['token_text'] = [
    [word for word in tweet.split() if word not in ['abortion', 'pro', 'life', 'choice', 'woman']]
    for tweet in df['clean_text']]
print(df['token_text'])"""

df['token_text'] = [
    [word for word in tweet.split()] for tweet in df['clean_text']]
print(df['token_text'])


#average cleaned tweet length for prolife and prochoice
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

#Average word length for prolife and prochoice
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


X_train, X_test, y_train, y_test = train_test_split(df['hashtags'], df['Unnamed: 8'])

#in order to use this we need to extract some features
#https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
#Above tutorial
#build a pipeline
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


MNB = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

from sklearn.linear_model import SGDClassifier
SGD = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])


#from sklearn import metrics
#print(metrics.classification_report(y_test, predicted))
#metrics.confusion_matrix(y_test, predicted)




#Do a parameter search
from sklearn.model_selection import GridSearchCV
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(SGD, parameters, cv=5, n_jobs=1)

gs_clf = gs_clf.fit(X_train, y_train)


gs_clf.best_score_

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

text = 'hashtags'

from sklearn.model_selection import cross_val_score, ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0) #Returns the same results each time
#cv = ShuffleSplit(n_splits=5, test_size=0.2) #returns slightly different results each time
scoresMNB = cross_val_score(MNB, df['hashtags'], df['Unnamed: 8'], cv=cv)
scoresMNB
np.mean(scoresMNB)


scoresSGD = cross_val_score(SGD, df['hashtags'], df['Unnamed: 8'], cv=cv)
scoresSGD
np.mean(scoresSGD)


print("Accuracy MNB: {:.2f}".format(np.mean(scoresMNB)))
print("Accuracy SGD: {:.2f}".format(np.mean(scoresSGD)))


#https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568

#Logistic regression
from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(n_jobs =1, C=1e5, solver = 'lbfgs', max_iter = 10000)),
])

scoreslogreg = cross_val_score(logreg, df['hashtags'], df['Unnamed: 8'], cv=cv)
scoreslogreg
np.mean(scoreslogreg)


print("Accuracy logistic regression: {:.2f}".format(np.mean(scoreslogreg)))


from sklearn.ensemble import RandomForestClassifier

ranfor = Pipeline([('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier(n_estimators=100)),
])




scoresMNBh = cross_val_score(MNB, df['hashtags'], df['Unnamed: 8'], cv=cv)
scoresSGDh = cross_val_score(SGD, df['hashtags'], df['Unnamed: 8'], cv=cv)
scoreslogregh = cross_val_score(logreg, df['hashtags'], df['Unnamed: 8'], cv=cv)
scoresranforh = cross_val_score(ranfor, df['hashtags'], df['Unnamed: 8'], cv=cv)
scoresMNBt = cross_val_score(MNB, df['clean_text'], df['Unnamed: 8'], cv=cv)
scoresSGDt = cross_val_score(SGD, df['clean_text'], df['Unnamed: 8'], cv=cv)
scoreslogregt = cross_val_score(logreg, df['clean_text'], df['Unnamed: 8'], cv=cv)
scoresranfort = cross_val_score(ranfor, df['clean_text'], df['Unnamed: 8'], cv=cv)
scoresMNBth = cross_val_score(MNB, df['clean_text_hashtag_text'], df['Unnamed: 8'], cv=cv)
scoresSGDth = cross_val_score(SGD, df['clean_text_hashtag_text'], df['Unnamed: 8'], cv=cv)
scoreslogregth = cross_val_score(logreg, df['clean_text_hashtag_text'], df['Unnamed: 8'], cv=cv)
scoresranforth = cross_val_score(ranfor, df['clean_text_hashtag_text'], df['Unnamed: 8'], cv=cv)


#Classification accuracy on hashtags, text, and both
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



scoresMNBlc = cross_val_score(MNB, df['lifeorchoice'], df['Unnamed: 8'], cv=cv)
scoresSGDlc = cross_val_score(SGD, df['lifeorchoice'], df['Unnamed: 8'], cv=cv)
scoreslogreglc = cross_val_score(logreg, df['lifeorchoice'], df['Unnamed: 8'], cv=cv)
scoresranforlc = cross_val_score(ranfor, df['lifeorchoice'], df['Unnamed: 8'], cv=cv)


print("\n\nClassification on only prolife or prochoice hashtags")
print("Accuracy Multinomial Naive Bayes: \t {:.2f}".format(np.mean(scoresMNBlc)))
print("Accuracy support vector machine: \t {:.2f}".format(np.mean(scoresSGDlc)))
print("Accuracy logistic regression: \t\t {:.2f}".format(np.mean(scoreslogreglc)))
print("Accuracy random forest classifier: \t {:.2f}".format(np.mean(scoresranforlc)))
























MNBtf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(use_idf = False)),
    ('clf', MultinomialNB())
])

MNBtfidf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

scoresMNBtf = cross_val_score(MNBtf, df['hashtags'], df['Unnamed: 8'], cv=cv)
scoresMNBtfidf = cross_val_score(MNBtfidf, df['hashtags'], df['Unnamed: 8'], cv=cv)



""" Okay, unfortunately we need to take our model out of the pipeline to view the most important features"""
#Attempts at extracting feature information


#http://ritchieng.com/machine-learning-multinomial-naive-bayes-vectorization/
vect = CountVectorizer()
vect.fit(X_train)

#Transform training data
X_train_dtm = vect.transform(X_train)

#Transform testing data into a document term matrix
X_test_dtm = vect.transform(X_test)

nb = MultinomialNB()

#train model
nb.fit(X_train_dtm, y_train)

#Make class predictions:
y_pred_class = nb.predict(X_test_dtm)

from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


X_train_tokens = vect.get_feature_names()
len(X_train_tokens)
print(X_train_tokens)

nb.feature_count_

# number of times each token appears across all prolife tweets
prolife_token_count = nb.feature_count_[0, :]
prolife_token_count


# number of times each token appears across all prochoice tweets
prochoice_token_count = nb.feature_count_[1, :]
prochoice_token_count

#Create a df representing tokens and their counts in each
tokens = pd.DataFrame({'token': X_train_tokens, 'prolife':prolife_token_count, 'prochoice':prochoice_token_count}).set_index('token')

tokens.head()

#Add 1 to each token to avoid dividing by zero
tokens['prolife'] = tokens.prolife + 1
tokens['prochoice'] = tokens.prochoice + 1

#Convert into frequencies
tokens['prolife'] = tokens.prolife / nb.class_count_[0]
tokens['prochoice'] = tokens.prochoice / nb.class_count_[1]

#Calculate ratio for each tweet
tokens['lifeRatio'] = tokens.prolife / tokens.prochoice


#Sort this by proportion
tokens.sort_values('lifeRatio', ascending=False)