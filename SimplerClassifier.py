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


for i in range(0,len(df)):
    hashtags = []
    for hashtag in df.iloc[i]['hashtags'].split():
        if hashtag.lower() == '#prolife':
            hashtags.append("prolife")
        if hashtag.lower() == '#prochoice':
            hashtags.append("prochoice")
    df.iloc[i, -1] = ' '.join([str(elem) for elem in hashtags])



df['Unnamed: 8'].value_counts()

#We create a new column with tokens
df['token_text'] = [
    [word for word in tweet.split() if word not in ['abortion', 'pro', 'life', 'choice', 'woman']]
    for tweet in df['clean_text']]
print(df['token_text'])

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
SVM = Pipeline([
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

gs_clf = GridSearchCV(SVM, parameters, cv=5, n_jobs=1)

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


scoresSVM = cross_val_score(SVM, df['hashtags'], df['Unnamed: 8'], cv=cv)
scoresSVM
np.mean(scoresSVM)


print("Accuracy MNB: {:.2f}".format(np.mean(scoresMNB)))
print("Accuracy SVM: {:.2f}".format(np.mean(scoresSVM)))


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
scoresSVMh = cross_val_score(SVM, df['hashtags'], df['Unnamed: 8'], cv=cv)
scoreslogregh = cross_val_score(logreg, df['hashtags'], df['Unnamed: 8'], cv=cv)
scoresranforh = cross_val_score(ranfor, df['hashtags'], df['Unnamed: 8'], cv=cv)
scoresMNBt = cross_val_score(MNB, df['clean_text'], df['Unnamed: 8'], cv=cv)
scoresSVMt = cross_val_score(SVM, df['clean_text'], df['Unnamed: 8'], cv=cv)
scoreslogregt = cross_val_score(logreg, df['clean_text'], df['Unnamed: 8'], cv=cv)
scoresranfort = cross_val_score(ranfor, df['clean_text'], df['Unnamed: 8'], cv=cv)
scoresMNBth = cross_val_score(MNB, df['clean_text_hashtag_text'], df['Unnamed: 8'], cv=cv)
scoresSVMth = cross_val_score(SVM, df['clean_text_hashtag_text'], df['Unnamed: 8'], cv=cv)
scoreslogregth = cross_val_score(logreg, df['clean_text_hashtag_text'], df['Unnamed: 8'], cv=cv)
scoresranforth = cross_val_score(ranfor, df['clean_text_hashtag_text'], df['Unnamed: 8'], cv=cv)


#Classification accuracy on hashtags, text, and both
print("Classification on hashtags")
print("Accuracy Multinomial Naive Bayes: \t {:.2f}".format(np.mean(scoresMNBh)))
print("Accuracy support vector machine: \t {:.2f}".format(np.mean(scoresSVMh)))
print("Accuracy logistic regression: \t\t {:.2f}".format(np.mean(scoreslogregh)))
print("Accuracy random forest classifier: \t {:.2f}".format(np.mean(scoresranforh)))

print("\n\nClassification on tweet text")
print("Accuracy Multinomial Naive Bayes: \t {:.2f}".format(np.mean(scoresMNBt)))
print("Accuracy support vector machine: \t {:.2f}".format(np.mean(scoresSVMt)))
print("Accuracy logistic regression: \t\t {:.2f}".format(np.mean(scoreslogregt)))
print("Accuracy random forest classifier: \t {:.2f}".format(np.mean(scoresranfort)))

print("\n\nClassification on hashtags and text")
print("Accuracy Multinomial Naive Bayes: \t {:.2f}".format(np.mean(scoresMNBth)))
print("Accuracy support vector machine: \t {:.2f}".format(np.mean(scoresSVMth)))
print("Accuracy logistic regression: \t\t {:.2f}".format(np.mean(scoreslogregth)))
print("Accuracy random forest classifier: \t {:.2f}".format(np.mean(scoresranforth)))



scoresMNBlc = cross_val_score(MNB, df['lifeorchoice'], df['Unnamed: 8'], cv=cv)
scoresSVMlc = cross_val_score(SVM, df['lifeorchoice'], df['Unnamed: 8'], cv=cv)
scoreslogreglc = cross_val_score(logreg, df['lifeorchoice'], df['Unnamed: 8'], cv=cv)
scoresranforlc = cross_val_score(ranfor, df['lifeorchoice'], df['Unnamed: 8'], cv=cv)


print("\n\nClassification on only prolife or prochoice hashtags")
print("Accuracy Multinomial Naive Bayes: \t {:.2f}".format(np.mean(scoresMNBlc)))
print("Accuracy support vector machine: \t {:.2f}".format(np.mean(scoresSVMlc)))
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







"""
def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))


print_top10(CountVectorizer, , df['hashtags'])

CountVectorizer(df['hashtags'])

CountVectorizer.get_feature_names"""