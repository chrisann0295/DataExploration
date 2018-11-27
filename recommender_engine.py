import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

data = pd.read_csv('google-play-store-apps/googleplaystore.csv')
#wait like 30s for this to finish
to_drop = ['Size',
          'Last Updated',
          'Current Ver',
          'Android Ver']
data.drop(to_drop, inplace=True, axis=1)
data.head()

def clean_data(data):
    # Fixing Price
    data = data.where(data['Price'] != "Everyone")
    data["Price"] = data["Price"].str.replace("$", '')
    data["Price"] = pd.to_numeric(data["Price"])

    # Fixing Installs
    data["Installs"] = data["Installs"].str.replace(",", '')
    data["Installs"] = data["Installs"].str.replace("+", '')
    data["Installs"] = pd.to_numeric(data["Installs"])

    data["Reviews"] = pd.to_numeric(data["Reviews"])
    data['Genres'] = data['Genres'].fillna('')
    return data

data = clean_data(data)
data.head()

def ranked_recommendations(recommended_data):
    C = recommended_data['Rating'].mean()
    print(C)

    # Calculate the minimum number of votes required to be in race (lower the more apps)
    m = recommended_data['Reviews'].quantile(0.20)
    print(m)

    # Filter out all qualified apps into a new DataFrame
    q_apps = recommended_data.copy().loc[data['Reviews'] >= m]
    q_apps.shape

    # Function that computes the weighted rating of each app
    def weighted_rating(x, m=m, C=C):
        v = x['Reviews']
        R = x['Rating']
        # Calculation based on the IMDB formula
        return (v/(v+m) * R) + (m/(m+v) * C)

    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    q_apps['score'] = q_apps.apply(weighted_rating, axis=1)

    q_apps = q_apps.sort_values('score', ascending=False)

    return q_apps


# GENRE BASED RECOMMENDER
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(data['Genres'])

tfidf_matrix.shape
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[0]

# We now have a pairwise cosine similarity matrix for all the apps in our dataset. 
# The next step is to write a function that returns the 30 most similar apps based on the cosine similarity score.
data = data.reset_index()
indices = pd.Series(data.index, index=data['App'])

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    new = data.iloc[movie_indices]
    return ranked_recommendations(new)

#  THESE RECOMMENDATIONS ARE BASED ON GENRES AND RANKED
print(get_recommendations('AI Draw | Art Filter for Selfie').head(40))