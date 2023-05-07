# %% [markdown]
# __Authors__: Bruno Javier Limon Avila, Carlo Volpe, Chiara Menchetti, Davide Di Virgilio
#
# ---
#
# ## Introduction
#
# The main focus of our analysis will be on fake news detection regarding the COVID-19 pandemic. The data comes from the following Kaggle dataset: [COVID-19 Fake News Detection](https://www.kaggle.com/datasets/arashnic/covid19-fake-news). There is a collection of 33 files, however not all of them are going to be used. The ones that will be considered are those containing news (which can be actual news from the web or tweets), disregarding those with claims.
#
# In this initial notebook it will be carried out the data loading and pre-processing of the data, which will result in a clean and complete dataset that will be used in the analysis that will follow.
#
# ## Data Loading
#
# The first task is to load the necessary raw files. To make this process easier and faster we have already combined some of the files present in the repository in the the following two:
#
# - __*news\_df*__ contains all the files that had news, namely: _NewsFakeCOVID-19.csv_, _NewsFakeCOVID-19\_5.csv_, _NewsFakeCOVID-19\_7.csv_, _NewsRealCOVID-19.csv_, _NewsRealCOVID-19\_5.csv_ and _NewsRealCOVID-19\_7.csv_.
#
# - __*tweets\_df*__ contains all the files that had news gathered from Twitter, namely: _NewsFakeCOVID-19\_tweets.csv_, _NewsFakeCOVID-19\_tweets\_5.csv_, _NewsFakeCOVID-19\_tweets\_7.csv_, _NewsRealCOVID-19\_tweets.csv_, _NewsRealCOVID-19\_tweets\_5.csv_ and _NewsRealCOVID-19\_tweets\_7.csv_.
#
# Regarding the __*news\_df*__, it was simply obtained by the concatenation of the previously reported files. The __*tweets\_df*__ was a more time consuming task since in the raw files was only present the ID number of the tweet, which required the user to perform theirselves a web-scrabing to get the actual content. The code that was used to get these datasets is commented below.

# %%
# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# Creation of news_dataset.csv

# # Files corresponding to news from the web
# web_data1 = pd.read_csv('NewsRealCOVID-19.csv')
# web_data2 = pd.read_csv('NewsRealCOVID-19_5.csv')
# web_data3 = pd.read_csv('NewsRealCOVID-19_7.csv')
# web_data4 = pd.read_csv('NewsFakeCOVID-19.csv')
# web_data5 = pd.read_csv('NewsFakeCOVID-19_5.csv')
# web_data6 = pd.read_csv('NewsFakeCOVID-19_7.csv')

# # Putting all web news together
# news_real = pd.concat([web_data1,web_data2,web_data3], axis =0)
# news_fake = pd.concat([web_data4,web_data5,web_data6], axis =0)

# news_real['Fake'] = 0
# news_fake['Fake'] = 1

# news_dataset = pd.concat([news_real, news_fake], axis=0)

# # Saving resulting file containing fake and real news
# news_dataset.to_csv('news_dataset.csv',index = False)

# %%
# # Creation of tweet_news_dataset.csv

# # Files corresponding to news from twitter, they contain an id which will be used to retrieve their content
# twt_data1 = pd.read_csv('NewsRealCOVID-19_tweets.csv')
# twt_data2 = pd.read_csv('NewsRealCOVID-19_tweets_5.csv')
# twt_data3 = pd.read_csv('NewsRealCOVID-19_tweets_7.csv')
# twt_data4 = pd.read_csv('NewsFakeCOVID-19_tweets.csv')
# twt_data5 = pd.read_csv('NewsFakeCOVID-19_tweets_5.csv')
# twt_data6 = pd.read_csv('NewsFakeCOVID-19_tweets_7.csv')

# # Putting all twitter ids together
# twt_news_real = pd.concat([twt_data1,twt_data2,twt_data3], axis = 0)
# twt_news_fake = pd.concat([twt_data4,twt_data5,twt_data6], axis = 0)

# # Adding labels
# twt_news_real['Fake'] = 0
# twt_news_fake['Fake'] = 1

# # Full dataset of twitter ids
# twt_news_dataset = pd.concat([twt_news_real, twt_news_fake], axis = 0)

# # Using tweepy to communicate with the twitter API and retrieve the content of the tweets based on the id provided
# # !pip install tweepy
# import tweepy
# client = tweepy.Client("AAAAAAAAAAAAAAAAAAAAAEKljAEAAAAAN%2FUO5WzSWGs8OcXpakqONnD99Js%3DufZ"
#                        +"i40hcECrvPzQKDbvphCk0imNiLpo7N2xfBqLtS1huN9jPjs", wait_on_rate_limit=True)

# # Creating a dictionary that indicates which twitter id is fake and which is real, to use later to assign correct label to content
# twt_ids = list(twt_news_dataset.iloc[:,1])
# twt_fake = list(twt_news_dataset.iloc[:,2])
# tweet_dictionary = dict(zip(twt_ids, twt_fake))

# # Using a for loop to fetch the tweets in batches of 100 ids to get the content using the tweepy client
# tweet_temp_content = []
# j = 100
# for i in range(0, 152000, 100):
#     get_tweet = client.get_tweets(twt_ids[i:j])
#     tweet_data = get_tweet.data
#     tweet_temp_content.append(tweet_data)
#     j = j+100

# # Flattening the results of the previous fetching to get all news in a single list
# tweet_flat = []
# for i in range(len(tweet_temp_content)):
#     tweet_flat += tweet_temp_content[i]

# # looping through the news list to single out possible empty values and assign them the appropriate values
# content = []
# content_id = []
# for i in range(len(tweet_flat)):
#     if tweet_flat[i]:
#         temp = tweet_flat[i].text
#         temp_id = tweet_flat[i].id
#         content.append(temp)
#         content_id.append(temp_id)
#     else:
#         temp = "NULL"
#         content.append(temp)

# # Creating a dataframe with the ids and content of the tweets
# tweet_news_dataset = pd.DataFrame(list(zip(content_id, content)), columns = ['tweet_id', 'tweet_content'])

# # Using the previous dictionary to assign the label to each record
# tweet_id = list(tweet_news_dataset.iloc[:,0])
# tweet_fake = []
# tweet_fake.extend(range(0, len(tweet_id)))
# for i in range(len(tweet_id)):
#     for key, value in tweet_dictionary.items():
#         if tweet_id[i] == key:
#             tweet_fake[i] = value

# # Add 'fake' column to indicate wether it's fake or real
# tweet_news_dataset['fake'] = tweet_fake

# # Saving final file
# tweet_news_dataset.to_csv('tweet_news_dataset.csv',index = False)

# %% [markdown]
# After creating these two datasets, it was necessary to combine them into one. For the news dataset only the headline of the articles is kept as content. So, the dataset will contain the content and the label variables. The label will be indicated as _fake_ and it will take 1 if the article/tweet is fake and 0 if it is instead real.

# %%
# # Code for combining the two datasets
#
# # Load the data
# news_df = pd.read_csv('news_dataset.csv')
# tweets_df = pd.read_csv('tweet_news_dataset.csv')
#
# # Select only the necessary variables from news_df and change their names
# news_df_new = news_df[['title', 'Fake']]
# news_df_new.columns = ['content', 'fake']
#
# # Select only the necessary variables from tweets_df and change their names
# tweets_df_new = tweets_df[['tweets_content', 'fake']]
# tweets_df_new.columns = ['content', 'fake']
#
# # Concatenate the two new dataframes
# df = pd.concat([news_df_new, tweets_df_new])
#
# # Save the dataframe into a .csv file
# df.to_csv('df_tweets_news_completo.csv', index=False)

# %% [markdown]
# In the zip folder of the material it has been included a data folder where you can find this complete dataset, namely __*df\_tweets\_news\_completo.csv*__. Having now the data into one file, the next step is the pre-processing.
#
# ## Data Pre-Processing
#
# Since the above codes may not be runned, the complete dataset in __*df\_tweets\_news\_completo.csv*__ can be loaded as a pandas dataframe, ready to be used.

# %%
# Load the complete dataset
df = pd.read_csv('df_tweets_news_completo.csv')
print(df.shape)
df.head()

# %% [markdown]
# As of now, in the dataset there are 129880 records between headlines and tweets.
#
# The following phase is to clean the dataset by converting all the text to lower case letters, removing links, mentions, emojis and other unnecessary characters.

# %%
# copy the dataset
df_clean = df.copy()

# convert to lower case letters and remove unnecessary information
df_clean['content'] = df['content'].str.lower()\
    .str.replace('(@[a-z0-9]+)\w+',' ')\
    .str.replace('(http\S+)', ' ')\
    .str.replace('(?::|;|=)(?:-)?(?:\)|\(|D|P)', ' ')\
    .str.replace('([^0-9a-z \t])', ' ')\
    .str.replace(' +',' ')

# %% [markdown]
# It may happen that data may contain duplicates, and since the data was not collected by us, it is normal procedure to check the content variable. And if there are any duplicates it is necessary to drop them.

# %%
# Check if there are any duplicates
duplicate = df_clean[df_clean.duplicated()]
print("Duplicate Rows :", duplicate.shape[0])

# %%
# Drop the duplicates
df_clean = df_clean.drop_duplicates()
print('The number of records in the dataset now is: ', df_clean.shape[0])

# %% [markdown]
# The number of the observations in the dataset has dropped by a lot by removing the duplicates.
#
# After this reduction, the data was checked manually to see if everything was okay. However, in the dataset at this points there are still some records having almost equal content. An example of this can be shown below, by printing some of the rows.

# %%
# Almost identical observations that were not caught as duplicates
df_clean['content'][85064:85069]

# %% [markdown]
# The idea to deal with this issue is to create a copy of the dataset but containing only the first 30 characters of the content variable. Then the duplicate function will be used again on this dataframe and the indexes of the this records will be removed from the original cleaned dataframe.
#
# To clarify, this might not be a problem (to have some almost identical records), however to make the classification more interesting it was decided to procede in this direction.

# %%
# Copy the dataframe
df_copy = df_clean.copy()

# Leave only the first 30 characters of the content variable
df_copy['content'] = df_copy['content'].str[0:30]

# Check for duplicates
duplicate2 = df_copy[df_copy.duplicated()]
print('Duplicate Rows: ', duplicate2.shape[0])

# Drop the duplicates
df_without_duplicates = df_clean.drop(index=duplicate2.index)
print('The number of records in the dataset now is: ', df_without_duplicates.shape[0])

# %% [markdown]
# The size of the dataset has shrunk some more, but at least the data is clean and without any duplicates. So, this data can be saved into a .csv file, to be exploited for the following analysis.

# %%
# Save in a .csv file
df_without_duplicates.to_csv(' df_without_duplicates.csv', index=False)

# %% [markdown]
# ## Data Understanding
#
# First of all it is necessary to check the proportion of real and fake news inside the pre-processed dataset.

# %%
df_without_duplicates['fake'].value_counts()

# %%
df_without_duplicates['fake'].value_counts(normalize=True)

# %%
df_without_duplicates['fake'].value_counts().plot(kind='pie')

# %% [markdown]
# The dataset is higly unbalanced (0: 64353 (0.97); 1: 2328 (0.03)). This should be taken into consideration when performing any supervised classification algorithm.
#
# The other variable in the dataframe, beside the label, is the text of the tweets and header of the article. To understand the content, the most frequent words can be plotted into a wordcloud. Before doing this the variable is tokenized and the stopwords are removed.

# %%
import nltk
nltk.download('stopwords')

# text cleaning
from nltk.corpus import stopwords

additional  = ['rt','rts','retweet']
swords = set().union(stopwords.words('english'),additional)

df_without_duplicates['processed_text'] = df_without_duplicates['content'].apply(lambda x: [i for i in x.split() if not i in swords])

# %%
# wordcloud of the whole dataset
from wordcloud import WordCloud, STOPWORDS

bigstring = df_without_duplicates['processed_text'].apply(lambda x: ' '.join(x)).str.cat(sep=' ')

plt.figure(figsize=(12,12))
wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          collocations=False,
                          width=1200,
                          height=1000
                         ).generate(bigstring)
plt.axis('off')
plt.imshow(wordcloud)

# %% [markdown]
# Since there are some very frequent words, which we already know are the main topic of the dataset, we can remove them and re-do the wordcloud.

# %%
# wordcloud without some words
stopwords_1 = ["coronavirus", "covid19","covid",'u'] + list(STOPWORDS)

plt.figure(figsize=(12,12))
wordcloud = WordCloud(stopwords=stopwords_1,
                          background_color='white',
                          collocations=False,
                          width=1200,
                          height=1000
                         ).generate(bigstring)
plt.axis('off')
plt.imshow(wordcloud)


