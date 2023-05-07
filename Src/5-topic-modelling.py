# %% [markdown]
# __authors__: Chiara Menchetti, Davide Di Virgilio, Carlo Volpe, Bruno Limon
#
# ___
#
# ## Topic Modeling

# %%
#libraries
import random
import numpy as np
import matplotlib as plt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import plotly.express as px

# %% [markdown]
# ## Topic Modeling on real news dataset

# %%
#loading real news dataset
df=pd.read_csv("df_without_duplicates.csv")
df_real=df[df["fake"]==0]
df_real

# %%
#text cleaning
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

en_stopwords = stopwords.words('english')
stopwords_1=["coronavirus", "covid19","covid",'u','19','de'] + list(en_stopwords)

# vectorization
tf_vectorizer = CountVectorizer(stop_words=stopwords_1, max_df=0.5, min_df=5,max_features = 7000, ngram_range=(1,2))

# Learn the vocabulary dictionary and return document-term matrix.
tf = tf_vectorizer.fit_transform(df_real["content"].values)


# %%
from sklearn.decomposition import LatentDirichletAllocation

n_components = 4

lda = LatentDirichletAllocation(n_components=n_components, max_iter=20,
                                learning_method = 'batch',
                                n_jobs=-1,verbose=1)
lda.fit(tf)


# %%
#function for printing n_top_words
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print()
        message = f'Topic {topic_idx}: '
        message += ', '.join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

# %%
#printing real news topics
n_top_words = 20
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

# %% [markdown]
# ## Visualization Phase

# %%
import pyLDAvis
import pyLDAvis.sklearn

# %%
pyLDAvis.enable_notebook()

# %%
pyLDAvis.sklearn.prepare(lda,tf,tf_vectorizer)

# %%
# Plot topics function
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(4,1, figsize=(10, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

# Show topics
n_top_words = 10
#feature_names = vectorizer_cv.get_feature_names_out()
plot_top_words(lda, tf_feature_names, n_top_words, '')

# %% [markdown]
# ## Topic Modeling on fake news dataset

# %%
#loading fake news dataset
df=pd.read_csv("df_without_duplicates.csv")
df_fake=df[df["fake"]==1]
df_fake

# %%
#Libraries
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

en_stopwords = stopwords.words('english')
stopwords_1=["coronavirus", "covid19","covid",'u','19','de'] + list(en_stopwords)

tf_vectorizer = CountVectorizer(stop_words=stopwords_1, max_df=0.5, min_df=5,max_features = 7000, ngram_range=(1,2))
tf_fake = tf_vectorizer.fit_transform(df_fake["content"].values)


# %%
from sklearn.decomposition import LatentDirichletAllocation

n_components_fake = 3

lda_fake = LatentDirichletAllocation(n_components=n_components, max_iter=20,
                                learning_method = 'batch',
                                n_jobs=-1,verbose=1)
lda_fake.fit(tf_fake)


# %%
n_top_words = 20
tf_feature_names_fake = tf_vectorizer.get_feature_names()
print_top_words(lda_fake, tf_feature_names_fake, n_top_words)

# %% [markdown]
# ## Visualization Phase

# %%
pyLDAvis.enable_notebook()

# %%
pyLDAvis.sklearn.prepare(lda_fake,tf_fake,tf_vectorizer)

# %%

# Plot topics function
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(3,1, figsize=(10, 10), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

# Show topics
n_top_words = 10
#feature_names = vectorizer_cv.get_feature_names_out()
plot_top_words(lda_fake, tf_feature_names_fake, n_top_words, '')


