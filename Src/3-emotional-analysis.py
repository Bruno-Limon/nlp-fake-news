# %% [markdown]
# ## DF Fake

# %% [markdown]
# ### DF Fake negative

# %%
#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns

# %%
df_fake_negative = pd.read_csv('df_fake_negative.csv')

# %%
df_fake_negative.head()

# %%
fakes_negative_without_neutral = df_fake_negative[df_fake_negative["emotion_2"]!='neutral']
fakes_negative_without_neutral.shape

# %%
fakes_negative_without_neutral["processed_text"]=fakes_negative_without_neutral["processed_text"].apply(lambda x: x.replace("'",""))

# %%
bigstring=fakes_negative_without_neutral["processed_text"].str.strip('[]').apply(lambda x: x.replace(",","")).str.cat(sep='')

# %%
# wordcloud of the fake dataset

from wordcloud import WordCloud, STOPWORDS

stopwords_1=["coronavirus", "covid19","covid",'u'] + list(STOPWORDS)


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
# ### DF Fake positive

# %%
df_fake_positive = pd.read_csv('df_fake_positive.csv')

# %%
fakes_positive_without_neutral = df_fake_positive[df_fake_positive["emotion_2"]!='neutral']
fakes_positive_without_neutral.shape

# %%
fakes_positive_without_neutral["processed_text"]=fakes_positive_without_neutral["processed_text"].apply(lambda x: x.replace("'",""))

# %%
fakes_positive_without_neutral

# %%
bigstring=fakes_positive_without_neutral["processed_text"].str.strip('[]').apply(lambda x: x.replace(",","")).str.cat(sep='')

# %%
# wordcloud of the fake dataset

from wordcloud import WordCloud, STOPWORDS

stopwords_1=["coronavirus", "covid19","covid",'u'] + list(STOPWORDS)


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
# ### DF Fake neutral

# %%
df_fake_neutral = pd.read_csv('df_fake_neutral.csv')

# %%
df_fake_neutral.head()

# %%
fakes_neutral_without_neutral = df_fake_neutral[df_fake_neutral["emotion_2"]!='neutral']
fakes_neutral_without_neutral.shape

# %%
fakes_neutral_without_neutral["processed_text"]=fakes_neutral_without_neutral["processed_text"].apply(lambda x: x.replace("'",""))

# %%
bigstring=fakes_neutral_without_neutral["processed_text"].str.strip('[]').apply(lambda x: x.replace(",","")).str.cat(sep='')

# %%
# wordcloud of the fake dataset

from wordcloud import WordCloud, STOPWORDS

stopwords_1=["coronavirus", "covid19","covid",'u'] + list(STOPWORDS)


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


# %% [markdown]
# ## DF Real

# %% [markdown]
# ### DF Real negative

# %%
df_real_negative = pd.read_csv('df_real_negative.csv')

# %%
real_negative_without_neutral = df_real_negative[df_real_negative["emotion_2"]!='neutral']
real_negative_without_neutral.shape

# %%
real_negative_without_neutral["processed_text"]=real_negative_without_neutral["processed_text"].apply(lambda x: x.replace("'",""))

# %%
bigstring=real_negative_without_neutral["processed_text"].str.strip('[]').apply(lambda x: x.replace(",","")).str.cat(sep='')

# %%
# wordcloud of the fake dataset

from wordcloud import WordCloud, STOPWORDS

stopwords_1=["coronavirus", "covid19","covid",'u'] + list(STOPWORDS)


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
# ### DF Real positive

# %%
df_real_positive = pd.read_csv('df_real_positive.csv')

# %%
real_positive_without_neutral = df_real_positive[df_real_positive["emotion_2"]!='neutral']
real_positive_without_neutral.shape

# %%
real_positive_without_neutral["processed_text"]=real_positive_without_neutral["processed_text"].apply(lambda x: x.replace("'",""))

# %%
bigstring=real_positive_without_neutral["processed_text"].str.strip('[]').apply(lambda x: x.replace(",","")).str.cat(sep='')

# %%
# wordcloud of the fake dataset

from wordcloud import WordCloud, STOPWORDS

stopwords_1=["coronavirus", "covid19","covid",'u'] + list(STOPWORDS)


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
# ### DF Real neutral

# %%
df_real_neutral = pd.read_csv('df_real_neutral.csv')

# %%
real_neutral_without_neutral = df_real_neutral[df_real_neutral["emotion_2"]!='neutral']
real_neutral_without_neutral.shape

# %%
real_neutral_without_neutral["processed_text"]=real_neutral_without_neutral["processed_text"].apply(lambda x: x.replace("'",""))

# %%
bigstring=real_neutral_without_neutral["processed_text"].str.strip('[]').apply(lambda x: x.replace(",","")).str.cat(sep='')

# %%
# wordcloud of the fake dataset

from wordcloud import WordCloud, STOPWORDS

stopwords_1=["coronavirus", "covid19","covid",'u'] + list(STOPWORDS)


plt.figure(figsize=(12,12))
wordcloud = WordCloud(stopwords=stopwords_1,
                          background_color='white',
                          collocations=False,
                          width=1200,
                          height=1000
                         ).generate(bigstring)
plt.axis('off')
plt.imshow(wordcloud)


