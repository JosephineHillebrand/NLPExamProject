#Check path:
"""
import sys
sys.path"""

#Packages
import pandas as pd
import numpy as np
from PIL import Image
from os import path
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
import nltk
from string import punctuation
from _collections import defaultdict
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import math


dflife = pd.read_csv("C:/Users/hille/Desktop/NLP/Project/data/#prolifesubsetannotated.csv")
dfchoice = pd.read_csv("C:/Users/hille/Desktop/NLP/Project/data/#prochoicesubsetannotated.csv")
dfabortion = pd.read_csv("C:/Users/hille/Desktop/NLP/Project/data/#abortionsubsetannotated.csv")

#Put together the dataframes
df = pd.concat([dflife, dfchoice, dfabortion], axis=0, ignore_index=True)

#Count number of each stance tweets
df['Unnamed: 8'].value_counts()
print(dflife['Unnamed: 8'].value_counts())
print("\n")
print(dfchoice['Unnamed: 8'].value_counts())
print(dfabortion['Unnamed: 8'].value_counts())



df1 = df

"""
lslife = df.loc[df['Unnamed: 8'] == -1.0]

#Get the different hashtags used in the different stances
wordcloud = WordCloud(max_font_size = 50, max_words = 100, background_color='white').generate(flattened_list)

liferow = lslife["hashtags"].tolist()
lifewords = [row.split(" ") for row in liferow]
flattened_list = [y for x in lifewords for y in x]
flattened_list = [re.sub("#", "", word) for word in flattened_list]
re.sub('#', '', flattened_list)

life = [row.split(' ') for row in liferow]
"""


text = df.hashtags[0]

wordcloud = WordCloud().generate(text)


plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#Wordcloud prolife tweets
hashtagsLife = " ".join(tweet for tweet in df.loc[df['Unnamed: 8'] == -1.0].hashtags)

wordcloudLifehashtags = WordCloud(background_color="white").generate(hashtagsLife)

plt.imshow(wordcloudLifehashtags, interpolation='bilinear')
plt.axis("off")
plt.show()

#Wordcloud prochoice tweets
hashtagsChoice = " ".join(tweet for tweet in df.loc[df['Unnamed: 8'] == 1.0].hashtags)

wordcloudChoicehashtags = WordCloud(background_color="white").generate(hashtagsChoice)

plt.imshow(wordcloudChoicehashtags, interpolation='bilinear')
plt.axis("off")
plt.show()

#Wordcloud neutral tweets
hashtagsNeutral = " ".join(tweet for tweet in df.loc[df['Unnamed: 8'] == 0].hashtags)

wordcloudNeutralhashtags = WordCloud(background_color="white").generate(hashtagsNeutral)

plt.imshow(wordcloudNeutralhashtags, interpolation='bilinear')
plt.axis("off")
plt.show()


#Wordcloud prolife text
textLife = " ".join(tweet for tweet in df.loc[df['Unnamed: 8'] == -1.0].text)

wordcloudLifetext = WordCloud(background_color="white").generate(textLife)

plt.imshow(wordcloudLifetext, interpolation='bilinear')
plt.axis("off")
plt.show()

#Wordcloud prochoice tweets
textChoice = " ".join(tweet for tweet in df.loc[df['Unnamed: 8'] == 1.0].text)

wordcloudChoicetext = WordCloud(background_color="white").generate(textChoice)

plt.imshow(wordcloudChoicetext, interpolation='bilinear')
plt.axis("off")
plt.show()

#Wordcloud neutral tweets
textNeutral = " ".join(tweet for tweet in df.loc[df['Unnamed: 8'] == 0].text)

wordcloudNeutraltext = WordCloud(background_color="white").generate(textNeutral)

plt.imshow(wordcloudNeutraltext, interpolation='bilinear')
plt.axis("off")
plt.show()


"""
#Clean the data
#Remove all tweets containing gun
cleanDF = df[~df.text.str.contains("gun")]
cleanDF = cleanDF[~df.text.str.contains("news")]
cleanDF = cleanDF[~df.text.str.contains("bizpac")]
cleanDF = cleanDF[~df.text.str.contains("segment")]
cleanDF = cleanDF[~df.text.str.contains("post")]
cleanDF = cleanDF[~df.text.str.contains("vacc")]

cleanDF = cleanDF[~df.hashtags.str.contains("2A")]

cleanDF['Unnamed: 8'].value_counts()
df['Unnamed: 8'].value_counts()

df = cleanDF
"""

df = df1


#We want to be able to remove URLs from our data
#We first define a function that finds URLs using regex
def find_URLs(tweets):
    return re.findall(r"((?:https?:\/\/(?:www\.)?|(?:pic\.|www\.)(?:\S*\.))(?:\S*))", tweets)

#We apply the function to our text column of our data frame
df['URLs'] = df.text.apply(find_URLs) 
df['URLs'].head(20)

#Then we can get rid of them inside of the tweet
df['clean_text'] = [re.sub(r"((?:https?:\/\/(?:www\.)?|(?:pic\.|www\.)(?:\S*\.))(?:\S*))",'', x) for x in df['text']]


#We start by defining lists of words to remove
my_stopwords = nltk.corpus.stopwords.words('english') #uninformative common words
my_punctuation = r'!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•…' #punctuation
#We specify the stemmer or lemmatizer we want to use
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
wordnet_lemmatizer = WordNetLemmatizer()


#We define our list so that we take out those words when tokenising
abortion = ['abortion']
woman = ["woman"]

#And we define a cleaning master function to do the heavy lifting
def clean_tweet(tweet, bigrams=False, lemma=False):
    tweet = tweet.lower() # lower case
    tweet = re.sub(r'#\S+', ' ', tweet) #Remove the hashtags from the text
    tweet = re.sub(r'@\S+', ' ', tweet) #Remove mentions
    tweet = re.sub(r'[^\w\s]', ' ', tweet) # strip punctuation
    tweet = re.sub(r'\s+', ' ', tweet) #remove double spacing
    tweet = re.sub(r'([0-9]+)', '', tweet) # remove numbers
    tweet = re.sub(r'([\U00002024-\U00002026]+)', '', tweet) #removing html tag ("..." where a link used to be)
    tweet_token_list = [word for word in tweet.split(' ')
                        if word not in my_stopwords and word not in abortion and word not in woman] #remove stopwords + custom stopwords

    if lemma == True:
      tweet_token_list = [wordnet_lemmatizer.lemmatize(word) if '#' not in word else word
                        for word in tweet_token_list] # apply lemmatizer
    else:   # or                 
      tweet_token_list = [word_rooter(word) if '#' not in word else word
                        for word in tweet_token_list] # apply word rooter
    if bigrams:
        tweet_token_list = tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1]
                                            for i in range(len(tweet_token_list)-1)]
    tweet = ' '.join(tweet_token_list)
    return tweet

#Finally we apply the function to clean tweets (here we use lemmas)
df['clean_text'] = df.clean_text.apply(clean_tweet, lemma=True)

df.head(20)

#df['clean_text_hashtag_text'] = df.clean_text.apply(clean_tweet, lemma=True)
#df.to_csv("dfClean.csv", index = False)

#Wordcloud prolife text
textLife = " ".join(tweet for tweet in df.loc[df['Unnamed: 8'] == -1.0].clean_text)

wordcloudLifetext = WordCloud(background_color="white", max_words = 20).generate(textLife)

plt.imshow(wordcloudLifetext, interpolation='bilinear')
plt.axis("off")
plt.show()

#Wordcloud prochoice tweets
textChoice = " ".join(tweet for tweet in df.loc[df['Unnamed: 8'] == 1.0].clean_text)

wordcloudChoicetext = WordCloud(background_color="white", max_words = 20).generate(textChoice)

plt.imshow(wordcloudChoicetext, interpolation='bilinear')
plt.axis("off")
plt.show()


#Wordcloud neutral tweets
textNeutral = " ".join(tweet for tweet in df.loc[df['Unnamed: 8'] == 0].clean_text)

wordcloudNeutraltext = WordCloud(background_color="white").generate(textNeutral)

plt.imshow(wordcloudNeutraltext, interpolation='bilinear')
plt.axis("off")
plt.show()






#####################################################################################
"""now that our data is cleaned, we want to start exploring a classifier on the text and the hashtags

We want to start with a Boolean Naive Bayes

We will only use positive or negative for starters and start with looking on the hashtags"""
#Thus, remove all zeroes
dfSub = df.loc[df['Unnamed: 8'].isin([-1.0,1.0])]
print(len(dfSub))


#Now we will build the Boolean NB model

totalTweets = len(dfSub)
probLife = math.log(len(df.loc[df['Unnamed: 8'] == -1.0])/totalTweets)
probChoice = math.log(len(df.loc[df['Unnamed: 8'] == 1.0])/totalTweets)



lifeWords = ' '.join(df.loc[df['Unnamed: 8'] == -1.0].hashtags)
lifeWords = lifeWords.lower()
choiceWords = ' '.join(df.loc[df['Unnamed: 8'] == 1.0].hashtags)
choiceWords = choiceWords.lower()

shared_vocab = set(lifeWords.split(" ")).intersection(set(choiceWords.split(" ")))
union_vocab = set(lifeWords.split(" ")).union(set(choiceWords.split(" ")))

vocab_size = len(union_vocab)

lifeProb = []
choiceProb = []


for hashtag in union_vocab:
    lifecount = lifeWords.count(hashtag)
    loglikelihoodlife = math.log((lifecount+1)/(len(lifeWords)+vocab_size))
    lifeProb.append(loglikelihoodlife)
    
    choicecount = choiceWords.count(hashtag)
    loglikelihoodchoice = math.log((choicecount+1)/(len(choiceWords)+vocab_size))
    choiceProb.append(loglikelihoodchoice)

#Put lists together
probability
probability = pd.DataFrame(
    {'hashtag': list(union_vocab),
     'probLife': lifeProb,
     'probChoice': choiceProb
    })


#now that our model is trained we want to predict on tweets

modelGuess = []
N = 0
for hashtags in df.hashtags:
    hashtags = hashtags.lower()
    #print(hashtags)

    hashtags = hashtags.split()

    #likelihood of life
    life_factors = 0
    #likelihood of choice
    choice_factors = 0

    for hashtag in hashtags:
        life = probability.loc[probability['hashtag'] == hashtag].probLife
        choice = probability.loc[probability['hashtag'] == hashtag].probChoice
        life_factors = life.values + life_factors
        
        choice_factors = choice.values + choice_factors
        

    if life_factors > choice_factors:
        modelGuess.append(-1.0)
    else:
        modelGuess.append(1.0)
    N = N+1
    #print(N)



#Add the modelGuess to our df

df['modelGuess'] = modelGuess


#Now check how many percent of the predictions are corrrect omitting neutral

dfSubModel = df.loc[df['Unnamed: 8'].isin([-1.0,1.0])]

correct = 0
wrong = 0
for i in range(0, len(dfSubModel)):
    if dfSubModel.iloc[i,8] == dfSubModel.iloc[i,16]:
        correct = correct+1
    else:
        wrong = wrong+1
    
#Correct 92.199 % of the time
correct/(wrong+correct)

wrong/(wrong+correct)


#how accurate is a model using only #Prolife or #prochoice
modelGuessSimple = []
N = 0
for hashtags in df.hashtags:
    hashtags = hashtags.lower()
    #print(hashtags)

    hashtags = hashtags.split()

    #likelihood of life
    life_factors = 0
    #likelihood of choice
    choice_factors = 0

    for hashtag in hashtags:
        if hashtag == "#prolife":
            life_factors = life_factors + 1
        if hashtag == "#prochoice":
            choice_factors = choice_factors + 1
        else:
            pass

    if life_factors > choice_factors:
        modelGuessSimple.append(-1.0)
    elif choice_factors > life_factors:
        modelGuessSimple.append(1.0)
    else:
        modelGuessSimple.append(0)
    N = N+1
    #print(N)

df['modelGuessSimple'] = modelGuessSimple



dfSubModelSimple = df.loc[df['Unnamed: 8'].isin([-1.0,1.0])]

correct = 0
wrong = 0
for i in range(0, len(dfSubModel)):
    if dfSubModelSimple.iloc[i,8] == dfSubModelSimple.iloc[i,17]:
        correct = correct+1
    else:
        wrong = wrong+1
    
#Correct 73.91 % of the time
correct/(wrong+correct)

wrong/(wrong+correct)



dfSubModelSimple = df.loc[df['Unnamed: 8'].isin([-1.0,1.0])]

correct = 0
wrong = 0
for i in range(0, len(dfSubModelSimple)):
    if dfSubModelSimple.iloc[i,8] == dfSubModelSimple.iloc[i,17]:
        correct = correct+1
    else:
        wrong = wrong+1
    
#Correct 73.91 % of the time
correct/(wrong+correct)

wrong/(wrong+correct)




############################################################################

############################################################################

############################################################################

############################################################################
#Try the NB model on the tweets instead


dfSub = df.loc[df['Unnamed: 8'].isin([-1.0,1.0])]
print(len(dfSub))


#Now we will build the Boolean NB model

totalTweets = len(dfSub)
probLife = math.log(len(dfSub.loc[dfSub['Unnamed: 8'] == -1.0])/totalTweets)
probChoice = math.log(len(dfSub.loc[dfSub['Unnamed: 8'] == 1.0])/totalTweets)



lifeWords = ' '.join(df.loc[df['Unnamed: 8'] == -1.0].clean_text)
lifeWords = lifeWords.lower()
choiceWords = ' '.join(df.loc[df['Unnamed: 8'] == 1.0].clean_text)
choiceWords = choiceWords.lower()

shared_vocab = set(lifeWords.split(" ")).intersection(set(choiceWords.split(" ")))
union_vocab = set(lifeWords.split(" ")).union(set(choiceWords.split(" ")))

vocab_size = len(union_vocab)

lifeProb = []
choiceProb = []


for word in union_vocab:
    lifecount = lifeWords.count(word)
    loglikelihoodlife = math.log((lifecount+1)/(len(lifeWords)+vocab_size))
    lifeProb.append(loglikelihoodlife)
    
    choicecount = choiceWords.count(word)
    loglikelihoodchoice = math.log((choicecount+1)/(len(choiceWords)+vocab_size))
    choiceProb.append(loglikelihoodchoice)

#Put lists together
probabilityWord = pd.DataFrame(
    {'word': list(union_vocab),
     'probLife': lifeProb,
     'probChoice': choiceProb
    })


#now that our model is trained we want to predict on tweets

modelGuess = []
N = 0
for tweet in df.clean_text:
    tweet = tweet.lower()
    #print(hashtags)

    tweet = tweet.split()

    #likelihood of life
    life_factors = 0
    #likelihood of choice
    choice_factors = 0

    for word in tweet:
        life = probabilityWord.loc[probabilityWord['word'] == word].probLife
        choice = probabilityWord.loc[probabilityWord['word'] == word].probChoice
        life_factors = life.values + life_factors
        
        choice_factors = choice.values + choice_factors
        

    if life_factors > choice_factors:
        modelGuess.append(-1.0)
        
    else:
        modelGuess.append(1.0)
        
    if life_factors == choice_factors:
        N = N+1
        



#Add the modelGuess to our df

df['modelTweets'] = modelGuess


#Now check how many percent of the predictions are corrrect omitting neutral

dfSubModelTweet = df.loc[df['Unnamed: 8'].isin([-1.0,1.0])]

correct = 0
wrong = 0
for i in range(0, len(dfSubModelTweet)):
    if dfSubModelTweet.iloc[i,8] == dfSubModelTweet.iloc[i,18]:
        correct = correct+1
    else:
        wrong = wrong+1
    
#Correct 50.895 % of the time
correct/(wrong+correct)

wrong/(wrong+correct)






#################################################################################

#################################################################################

#################################################################################

#################################################################################

#################################################################################
#Now we want to try and label both pos, neg and neu tweets


#Now we will build the Boolean NB model

totalTweets = len(df)
probLife = math.log(len(df.loc[df['Unnamed: 8'] == -1.0])/totalTweets)
probChoice = math.log(len(df.loc[df['Unnamed: 8'] == 1.0])/totalTweets)
probNeu = math.log(len(df.loc[df['Unnamed: 8'] == 0])/totalTweets)


lifeWords = ' '.join(df.loc[df['Unnamed: 8'] == -1.0].hashtags)
lifeWords = lifeWords.lower()
choiceWords = ' '.join(df.loc[df['Unnamed: 8'] == 1.0].hashtags)
choiceWords = choiceWords.lower()
neuWords = ' '.join(df.loc[df['Unnamed: 8'] == 0].hashtags)
choiceWords = choiceWords.lower()

shared_vocab = set(lifeWords.split(" ")).intersection(set(choiceWords.split(" ")))
shared_vocab = shared_vocab.intersection(set(neuWords.split(" ")))
union_vocab = set(lifeWords.split(" ")).union(set(choiceWords.split(" ")))
union_vocab = union_vocab.union(set(neuWords.split(" ")))

vocab_size = len(union_vocab)

lifeProb = []
choiceProb = []
neuProb = []

for hashtag in union_vocab:
    lifecount = lifeWords.count(hashtag)
    loglikelihoodlife = math.log((lifecount+1)/(len(lifeWords)+vocab_size))
    lifeProb.append(loglikelihoodlife)
    
    choicecount = choiceWords.count(hashtag)
    loglikelihoodchoice = math.log((choicecount+1)/(len(choiceWords)+vocab_size))
    choiceProb.append(loglikelihoodchoice)

    neucount = neuWords.count(hashtag)
    loglikelihoodneu = math.log((neucount+1)/(len(neuWords)+vocab_size))
    neuProb.append(loglikelihoodneu)

#Put lists together

probability = pd.DataFrame(
    {'hashtag': list(union_vocab),
     'probLife': lifeProb,
     'probChoice': choiceProb,
     'probNeu': neuProb
    })


#now that our model is trained we want to predict on tweets

modelGuess = []
N = 0
for hashtags in df.hashtags:
    hashtags = hashtags.lower()
    #print(hashtags)

    hashtags = hashtags.split()

    #likelihood of life
    life_factors = 0
    #likelihood of choice
    choice_factors = 0

    #likelihood of neu
    neu_factors = 0

    for hashtag in hashtags:
        life = probability.loc[probability['hashtag'] == hashtag].probLife
        choice = probability.loc[probability['hashtag'] == hashtag].probChoice
        neu = probability.loc[probability['hashtag'] == hashtag].probNeu
        life_factors = life.values + life_factors
        
        choice_factors = choice.values + choice_factors
        
        neu_factors = neu.values + neu_factors

    if life_factors > choice_factors:
        if neu_factors > life_factors:
            modelGuess.append(0)
        else:
            modelGuess.append(-1.0)
    elif choice_factors > neu_factors:
        modelGuess.append(1.0)
    else:
        modelGuess.append(0)




#Add the modelGuess to our df

df['modelGuessAllCategories'] = modelGuess


#Now check how many percent of the predictions are corrrect omitting neutral

correct = 0
wrong = 0
for i in range(0, len(df)):
    if df.iloc[i,8] == df.iloc[i,19]:
        correct = correct+1
    else:
        wrong = wrong+1
    
#Correct 82.4 % of the time
correct/(wrong+correct)

wrong/(wrong+correct)

