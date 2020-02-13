#Check path:
"""
import sys
sys.path"""

"""
This document contains the code used for cleaning up the data
"""

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

#Read in the 3 different dataframes of the random subsets of each dataframe.  i.e. in this case we have 400 tweets in each dataframe, which has been manually annotated
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
Now we want to clean the data.
This code has been greatly inspired by code presented by Rebekah Baglini, Luca Nannini & Arnault-Quentin Vermillet
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
