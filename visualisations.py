#Packages
from PIL import Image
from os import path
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%reset - a command to remove all variables/reset them

#Read in the data
df = pd.read_csv("C:/Users/hille/Desktop/NLP/Project/dfClean.csv")


#Let us just look at how many similar tweets are there (after removing hashtags and stuff)


dfRemoveDuplicate = df.copy()
dfRemoveDuplicate = dfRemoveDuplicate.drop_duplicates(subset=['clean_text'], keep='first')

#Now let us make wordclouds for prolife, prochoice and neutral tweets

babyMask = np.array(Image.open("C:/Users/hille/Desktop/NLP/Project/pregnant.png"))


def transform_format(val):
    if val == 0:
        return 255
    else:
        return val

# Transform your mask into a new one that will work with the function:
transformed_mask = np.ndarray((babyMask.shape[0],babyMask.shape[1]), np.int32)

for i in range(len(babyMask)):
    transformed_mask[i] = list(map(transform_format, babyMask[i]))


#Wordcloud prolife text
textLife = " ".join(str(tweet) for tweet in dfRemoveDuplicate.loc[dfRemoveDuplicate['Unnamed: 8'] == -1.0].clean_text)
#Wordcloud in specific shape
wordcloudLifetext = WordCloud(max_words = 50, background_color="white", stopwords = ["abortion", 'woman'], mask = transformed_mask, contour_color = "black", contour_width = 3).generate(textLife)

fig, ax = plt.subplots()
plt.imshow(wordcloudLifetext, interpolation='bilinear')
plt.axis("off")
plt.show()

#Wordcloud in normal square shape
wordcloudLifetext = WordCloud(max_words = 50, background_color="white", stopwords = ["abortion", 'woman'], contour_color = "black", contour_width = 3).generate(textLife)





#fig.savefig('babyimage4.png', format='png', dpi=2400)



#Wordcloud prochoice text
###################################

womanMask = np.array(Image.open("C:/Users/hille/Desktop/NLP/Project/woman.png"))


def transform_format(val):
    if val == 0:
        return 255
    else:
        return val

# Transform your mask into a new one that will work with the function:
woman_mask = np.ndarray((womanMask.shape[0],womanMask.shape[1]), np.int32)

for i in range(len(womanMask)):
    woman_mask[i] = list(map(transform_format, womanMask[i]))


textChoice = " ".join(str(tweet) for tweet in dfRemoveDuplicate.loc[dfRemoveDuplicate['Unnamed: 8'] == 1.0].clean_text)

#Wordcloud in the shape of a woman
wordcloudChoiceText = WordCloud(max_words = 50, background_color="white", stopwords = ["abortion", 'woman'], mask = woman_mask, contour_color = "black", contour_width = 3).generate(textChoice)

#Wordcloud in normal square shape
wordcloudChoiceText = WordCloud(max_words = 50, background_color="white", stopwords = ["abortion", 'woman'], contour_color = "black", contour_width = 3).generate(textChoice)


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plt.imshow(wordcloudChoiceText, interpolation='bilinear')
plt.axis("off")
plt.show()




# Do the plot code
#fig.savefig('prochoice1.png', format='png', dpi=2400)

#fig.savefig('babyimage2.svg', format='svg', dpi=10000)

