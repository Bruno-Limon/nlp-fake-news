# %% [markdown]
# __Authors__: Bruno Javier Limon Avila, Carlo Volpe, Chiara Menchetti, Davide Di Virgilio
#
# ---
#
# # Sentiment Analysis
#
# This section revolves around Sentiment analysis, in an unsupervised way.

# %%
#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from my_utils import *
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS

# %% [markdown]
# ## Data loading
# Load the cleaned dataframe (computed in the jupyter notebook `0_TXA_data_preprocessing.ipynb`).

# %%
#loading dataset
df=pd.read_csv("df_without_duplicates.csv")

# %%
# text cleaning and tokenization
additional  = ['rt','rts','retweet']
swords = set().union(stopwords.words('english'),additional)

df['processed_text'] = df['content'].str.lower()\
          .str.replace('(@[a-z0-9]+)\w+',' ')\
          .str.replace('(http\S+)', ' ')\
          .str.replace('([^0-9a-z \t])',' ')\
          .str.replace(' +',' ')\
          .apply(lambda x: [i for i in x.split() if not i in swords])

# %%
df.info()

# %%
df

# %% [markdown]
# Our goal is to perform Sentiment Analysis on fake and reals news separately.

# %%
# splitting in two the dataset
df_fake=df[df["fake"]==1]
df_real=df[df["fake"]==0]

# %% [markdown]
# For a bettere visualization we decided to not consider the followings words: "coronavirus", "covid19", "covid", "u"

# %%
# wordcloud of the fake dataset
stopwords_1=["coronavirus", "covid19","covid",'u'] + list(STOPWORDS)
bigstring = df_fake['processed_text'].apply(lambda x: ' '.join(x)).str.cat(sep=' ')


plt.figure(figsize=(12,12))
wordcloud = WordCloud(stopwords=stopwords_1,
                          background_color='white',
                          collocations=False,
                          width=1200,
                          height=1000
                         ).generate(bigstring)
plt.axis('off')
plt.imshow(wordcloud)

# %%
# wordcloud of the trues dataset
stopwords_1=["coronavirus", "covid19","covid",'u'] + list(STOPWORDS)
bigstring = df_real['processed_text'].apply(lambda x: ' '.join(x)).str.cat(sep=' ')


plt.figure(figsize=(12,12))
wordcloud = WordCloud(stopwords=stopwords_1,
                          background_color='white',
                          collocations=False,
                          width=1200,
                          height=1000
                         ).generate(bigstring)
plt.axis('off')
plt.imshow(wordcloud)

# %% [markdown]
# <a id="fakenews"></a>
# ## Fake News Sentiment Analysis

# %% [markdown]
# In this subsection it is performed the sentiment analysis on the dataset only containing fake news and tweets.  The first step is the stemming:

# %%
# stemmization
from nltk.stem import PorterStemmer
ps = PorterStemmer()
df_fake['stemmer'] = df_fake['processed_text'].apply(lambda x: [ps.stem(i) for i in x if i != ''])

# %%
df_fake.head(3)

# %% [markdown]
# To perform the sentiment analysis as an unsupervised task, it is used the Vader Lexicon.

# %%
# sentiment analysis
import nltk.sentiment.vader as vd
from nltk import download
download('vader_lexicon')
sia = vd.SentimentIntensityAnalyzer()

# %%
#tokenization and sentiment
from nltk.tokenize import word_tokenize
df_fake['sentiment_score'] = df_fake['stemmer'].apply(lambda x: sum([ sia.polarity_scores(i)['compound'] for i in word_tokenize( ' '.join(x) )]) )

# %%
# number of positives, neutrals and negatives
print('positives: ', len(df_fake[df_fake["sentiment_score"]>0]))
print('neutrals: ', len(df_fake[df_fake["sentiment_score"]==0]))
print('negatives: ', len(df_fake[df_fake["sentiment_score"]<0]))

# %% [markdown]
# To further the investigation into the connotation of the records in the dataset, the idea is to perform emotion analysis, for each of the three sentiments separately.

# %%
# create the three datasets divided by the sentiment score: positive, negative and neutral
df_fake_positive = df_fake[df_fake["sentiment_score"]>0]
df_fake_neutral = df_fake[df_fake["sentiment_score"]==0]
df_fake_negative = df_fake[df_fake["sentiment_score"]<0]

# %% [markdown]
# ### Negatives Emotion Analysis
#
# For the emotion analysis the algorithms used are two. The first one is the following, by using the NRCLex function:

# %%
from nrclex import NRCLex

# %%
df_fake_negative['emotions'] = df_fake_negative['content'].apply(lambda x: NRCLex(x).affect_frequencies)
df_fake_negative.head(3)

# %%
# create a df with only the content and the emotion
fakes_negative_emotions = pd.concat([df_fake_negative.content, df_fake_negative['emotions'].apply(pd.Series)], axis = 1)
fakes_negative_emotions = fakes_negative_emotions.drop(['anticip'], axis=1)
print(fakes_negative_emotions.shape)
fakes_negative_emotions.head(3)

# %% [markdown]
# As it can be observed, this library assigns a score to each emotion for each record. There can also be cases in which all the emotions are set to zero. To gather the approximate emotions for this dataset, the average of each emotion is computed.

# %%
# mean value for each emotion
np.mean(fakes_negative_emotions)

# %% [markdown]
# The highest emotion is the *negative* one, followed by the *positive* and *anticipation* ones.

# %% [markdown]
# The next algorithm exploited is built upon the transformers library, and it uses a model called *EmoRoBERTa*.

# %%
from transformers import pipeline

# %%
emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

# %%
def get_emotion_label(text):
  return(emotion(text)[0]['label'])

# %% [markdown]
# The following is a time-consuming cell
#
# The computation of this algorithm tend to be very slow. If the the file containing the result of this operations is available, `run_time_consuming_ops = False` can be used, and the time-consuming operations won't be executed, since the cell will simply load the result file to save time.
# If you want to run the time-consuming operations anyway, set `run_time_consuming_ops = True`.

# %%
# computing the emotion for the negatives dataset
run_time_consuming_ops = False

if run_time_consuming_ops:

    df_fake_negative['emotion_2'] = df_fake_negative['content'].apply(get_emotion_label)
    print('Time consuming operations performed!')

else:

    df_fake_negative = pd.read_csv('df_fake_negative.csv')
    print('Time-saving file loaded!')

# %%
df_fake_negative.head(3)

# %%
df_fake_negative['processed_text'] = df_fake_negative['processed_text'].apply(lambda x: x.replace("'", ""))

# %%
bigstring = df_fake_negative['processed_text'].apply(lambda x: ''.join(x)).str.cat(sep='')
bigstring

# %% [markdown]
# Each observation (headline or tweet) is assigned an emotion. Let's plot the emotions.

# %%
sns.countplot(data = df_fake_negative, y = 'emotion_2').set(title = "Emotion Distribution", ylabel= "Emotion")

# %% [markdown]
# There is a lot of neutrally connotated content. To look closely at the other emotions, the neutral ones will be removed from the plot.

# %%
fakes_negative_without_neutral = df_fake_negative[df_fake_negative["emotion_2"]!='neutral']
fakes_negative_without_neutral.shape

# %%
sns.countplot(data = fakes_negative_without_neutral, y = 'emotion_2').set(ylabel= "Emotion")

# %% [markdown]
# ### Positives Emotion Analysis
#
# First the NRCLex function emotion will be exploited.

# %%
df_fake_positive['emotions'] = df_fake_positive['content'].apply(lambda x: NRCLex(x).affect_frequencies)
df_fake_positive.head(3)

# %%
# create a df with only the content and the emotion
fakes_positive_emotions = pd.concat([df_fake_positive.content, df_fake_positive['emotions'].apply(pd.Series)], axis = 1)
fakes_positive_emotions = fakes_positive_emotions.drop(['anticip'], axis=1)
print(fakes_positive_emotions.shape)
fakes_positive_emotions.head(3)

# %%
# mean value for each emotion
np.mean(fakes_positive_emotions)

# %% [markdown]
# The emotion with the highest average score is the *positive* one, followed by the *anticipation*, *negative* and *trust* ones.
#
# Then the other algorithm using transformers is exploited.
#
# The following is a time-consuming cell ($\approx$ 2 min).

# %%
# computing the emotion for the positives dataset
run_time_consuming_ops = False

if run_time_consuming_ops:

    df_fake_positive['emotion_2'] = df_fake_positive['content'].apply(get_emotion_label)
    print('Time consuming operations performed!')

else:

    df_fake_positive = pd.read_csv('df_fake_positive.csv')
    print('Time-saving file loaded!')

# %%
df_fake_positive.head(3)

# %%
sns.countplot(data = df_fake_positive, y = 'emotion_2').set(title = "Emotion Distribution", ylabel= "Emotion")

# %% [markdown]
# There is a lot of neutrally connotated content. To look closely at the other emotions, the neutral ones will be removed from the plot.

# %%
fakes_positive_without_neutral = df_fake_positive[df_fake_positive["emotion_2"]!='neutral']
fakes_positive_without_neutral.shape

# %%
sns.countplot(data = fakes_positive_without_neutral, y = 'emotion_2').set(ylabel= "Emotion")

# %% [markdown]
# ### Neutrals Emotion Analysis
# First the NRCLex function emotion will be exploited.

# %%
df_fake_neutral['emotions'] = df_fake_neutral['content'].apply(lambda x: NRCLex(x).affect_frequencies)
df_fake_neutral.head(3)

# %%
# create a df with only the content and the emotion
fakes_neutral_emotions = pd.concat([df_fake_neutral.content, df_fake_neutral['emotions'].apply(pd.Series)], axis = 1)
fakes_neutral_emotions = fakes_neutral_emotions.drop(['anticip'], axis=1)
print(fakes_neutral_emotions.shape)
fakes_neutral_emotions.head(3)

# %%
# mean value for each emotion
np.mean(fakes_neutral_emotions)

# %% [markdown]
# The emotions with highest average are *positive* and *anticipation*.
#
# Then the other algorithm using transformers is exploited.
#
# The following is a time-consuming cell ($\approx$ 2 min).

# %%
# computing the emotion for the negatives dataset
run_time_consuming_ops = False

if run_time_consuming_ops:

    df_fake_neutral['emotion_2'] = df_fake_neutral['content'].apply(get_emotion_label)
    print('Time consuming operations performed!')

else:

    df_fake_neutral = pd.read_csv('df_fake_neutral.csv')
    print('Time-saving file loaded!')

# %%
df_fake_neutral.head(3)

# %%
sns.countplot(data = df_fake_neutral, y = 'emotion_2').set(title = "Emotion Distribution", ylabel= "Emotion")

# %% [markdown]
# There is a lot of neutrally connotated content. To look closely at the other emotions, the neutral ones will be removed from the plot.

# %%
fakes_neutral_without_neutral = df_fake_neutral[df_fake_neutral["emotion_2"]!='neutral']
fakes_neutral_without_neutral.shape

# %%
sns.countplot(data = fakes_neutral_without_neutral, y = 'emotion_2').set( ylabel= "Emotion")

# %% [markdown]
# ## Real News Sentiment Analysis
#
# In this subsection it is performed the sentiment analysis on the dataset only containing real news and tweets.  The first step is the stemming:

# %%
# stemmization
from nltk.stem import PorterStemmer
ps = PorterStemmer()
df_real['stemmer'] = df_real['processed_text'].apply(lambda x: [ps.stem(i) for i in x if i != ''])

# %% [markdown]
# To perform the sentiment analysis as an unsupervised task, it is used the Vader Lexicon.

# %%
from nltk.tokenize import word_tokenize
df_real['sentiment_score'] = df_real['stemmer'].apply(lambda x: sum([ sia.polarity_scores(i)['compound'] for i in word_tokenize( ' '.join(x) )]) )

# %%
# number of positives, neutrals and negatives
print('positives: ', len(df_real[df_real["sentiment_score"]>0]))
print('neutrals: ', len(df_real[df_real["sentiment_score"]==0]))
print('negatives: ', len(df_real[df_real["sentiment_score"]<0]))

# %% [markdown]
# To further the investigation into the connotation of the records in the dataset, the idea is to perform emotion analysis, for each of the three sentiments separately.

# %%
# create the three datasets divided by the sentiment score: positive, negative and neutral
df_real_positive = df_real[df_real["sentiment_score"]>0]
df_real_neutral = df_real[df_real["sentiment_score"]==0]
df_real_negative = df_real[df_real["sentiment_score"]<0]

# %% [markdown]
# ### Negatives Emotion Analysis
#
# For the emotion analysis the algorithms used are two. The first one is the following, by using the NRCLex function:

# %%
df_real_negative['emotions'] = df_real_negative['content'].apply(lambda x: NRCLex(x).affect_frequencies)
df_real_negative.head(3)

# %%
# create a df with only the content and the emotion
reals_negative_emotions = pd.concat([df_real_negative.content, df_real_negative['emotions'].apply(pd.Series)], axis = 1)
reals_negative_emotions = reals_negative_emotions.drop(['anticip'], axis=1)
print(reals_negative_emotions.shape)
reals_negative_emotions.head(3)

# %%
# mean value for each emotion
np.mean(reals_negative_emotions)

# %% [markdown]
# The highest emotion on average is the *negative* one, followed by the *positive*, *anticipation*, *fear* and *trust* ones.
#
# The next algorithm exploited is built upon the transformers library, and it uses a model called *EmoRoBERTa*.
#
# The following is a time-consuming cell ($\approx$ 45 min).
#
# The computation of this algorithm tends to be very slow. If the the file containing the result of this operations is available, `run_time_consuming_ops = False` can be used, and the time-consuming operations won't be executed, since the cell will simply load the result file to save time.
# If you want to run the time-consuming operations anyway, set `run_time_consuming_ops = True`.

# %%
# computing the emotion for the negatives dataset
run_time_consuming_ops = False

if run_time_consuming_ops:

    df_real_negative['emotion_2'] = df_real_negative['content'].apply(get_emotion_label)
    print('Time consuming operations performed!')

else:

    df_real_negative = pd.read_csv('df_real_negative.csv')
    print('Time-saving file loaded!')

# %%
df_real_negative.head(3)

# %%
sns.countplot(data = df_real_negative, y = 'emotion_2').set(title = "Emotion Distribution", ylabel= "Emotion")

# %% [markdown]
# There is a lot of neutrally connotated content. To look closely at the other emotions, the neutral ones will be removed from the plot.

# %%
reals_negative_without_neutral = df_real_negative[df_real_negative["emotion_2"]!='neutral']
reals_negative_without_neutral.shape

# %%
sns.countplot(data = reals_negative_without_neutral, y = 'emotion_2').set(ylabel= "Emotion")

# %% [markdown]
# ### Positives Emotion Analysis
# First the NRCLex function emotion will be exploited.

# %%
df_real_positive['emotions'] = df_real_positive['content'].apply(lambda x: NRCLex(x).affect_frequencies)
df_real_positive.head(3)

# %%
# create a df with only the content and the emotion
reals_positive_emotions = pd.concat([df_real_positive.content, df_real_positive['emotions'].apply(pd.Series)], axis = 1)
reals_positive_emotions = reals_positive_emotions.drop(['anticip'], axis=1)
print(reals_positive_emotions.shape)
reals_positive_emotions.head(3)

# %%
# mean value for each emotion
np.mean(reals_positive_emotions)

# %% [markdown]
# The highest emotions on average are *positive*, *trust* and *anticipation*.
#
# Then the other algorithm using transformers is exploited.
#
# The following is a time-consuming cell ($\approx$ 65 min).

# %%
# computing the emotion for the positives dataset
run_time_consuming_ops = False

if run_time_consuming_ops:

    df_real_positive['emotion_2'] = df_real_positive['content'].apply(get_emotion_label)
    print('Time consuming operations performed!')

else:

    df_real_positive = pd.read_csv('df_real_positive.csv')
    print('Time-saving file loaded!')

# %%
df_real_positive.head(3)

# %%
sns.countplot(data = df_real_positive, y = 'emotion_2').set(title = "Emotion Distribution", ylabel= "Emotion")

# %% [markdown]
# There is a lot of neutrally connotated content. To look closely at the other emotions, the neutral ones will be removed from the plot.

# %%
reals_positive_without_neutral = df_real_positive[df_real_positive["emotion_2"]!='neutral']
reals_positive_without_neutral.shape

# %%
sns.countplot(data = reals_positive_without_neutral, y = 'emotion_2').set( ylabel= "Emotion")

# %% [markdown]
# ### Neutrals Emotion Analysis
# First the NRCLex function emotion will be exploited.

# %%
df_real_neutral['emotions'] = df_real_neutral['content'].apply(lambda x: NRCLex(x).affect_frequencies)
df_real_neutral.head(3)

# %%
# create a df with only the content and the emotion
reals_neutral_emotions = pd.concat([df_real_neutral.content, df_real_neutral['emotions'].apply(pd.Series)], axis = 1)
reals_neutral_emotions = reals_neutral_emotions.drop(['anticip'], axis=1)
print(reals_neutral_emotions.shape)
reals_neutral_emotions.head(3)

# %%
# mean value for each emotion
np.mean(reals_neutral_emotions)

# %% [markdown]
# The highest emotions on average are *positive*, *anticipation* and *trust*.
#
# Then the other algorithm using transformers is exploited.
#
# The following is a time-consuming cell ($\approx$ 30 min).

# %%
# computing the emotion for the neutrals dataset
run_time_consuming_ops = False

if run_time_consuming_ops:

    df_real_neutral['emotion_2'] = df_real_neutral['content'].apply(get_emotion_label)
    print('Time consuming operations performed!')

else:

    df_real_neutral = pd.read_csv('df_real_neutral.csv')
    print('Time-saving file loaded!')

# %%
df_real_neutral.head(3)

# %%
sns.countplot(data = df_real_neutral, y = 'emotion_2').set(title = "Emotion Distribution", ylabel= "Emotion")

# %% [markdown]
# There is a lot of neutrally connotated content. To look closely at the other emotions, the neutral ones will be removed from the plot.

# %%
reals_neutral_without_neutral = df_real_neutral[df_real_neutral["emotion_2"]!='neutral']
reals_neutral_without_neutral.shape

# %%
sns.countplot(data = reals_neutral_without_neutral, y = 'emotion_2').set(ylabel= "Emotion")


