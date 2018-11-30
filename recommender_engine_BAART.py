import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

data = pd.read_csv('data/googleplaystore.csv')
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

    # Fix Reviews
    data["Reviews"] = pd.to_numeric(data["Reviews"])

    #Fix Ratings
    data = data.where(data['Rating'] <= 5.0)

    # Fix Genres
    data['Genres'] = data['Genres'].fillna('')
    return data

data = clean_data(data)
# data.head()

def ranked_recommendations(recommended_data):
    C = recommended_data['Rating'].mean()
    # print(C)

    # Calculate the minimum number of votes required to be in race (lower the more apps)
    m = recommended_data['Reviews'].quantile(0.20)
    # print(m)

    # Filter out all qualified apps into a new DataFrame
    q_apps = recommended_data.copy().loc[data['Reviews'] >= m]

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

def get_similarities(data, alg):
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    # tf = TfidfVectorizer(analyzer='word', stop_words='english')
    tfidf_matrix = tf.fit_transform(data['Genres'])
    # tfidf_matrix = tf.fit_transform(data['App'])
    print('Unique Genres')
    print(data.Genres.unique())
    print(data.Genres.nunique())
    print('tfidf matrix')
    print(tfidf_matrix)
    print(tfidf_matrix.shape)
    if alg == 'cosine':
        return linear_kernel(tfidf_matrix, tfidf_matrix)
    elif alg in ['pearson', 'kendall', 'spearman']:
        return 
    else:
        print('Similarity algorithm unknown. Please select from the following options:\n - cosine\n - pearson\n - kendall\n - spearman')
        exit()

# # GENRE BASED RECOMMENDER
# tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
# tfidf_matrix = tf.fit_transform(data['Genres'])

# TITLE BASED RECCOMENDER
# translation_table = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', \
#              6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', \
#             11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', \
#             15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', \
#             19: 'Nineteen', 20: 'Twenty', 30: 'Thirty', 40: 'Forty', \
#             50: 'Fifty', 60: 'Sixty', 70: 'Seventy', 80: 'Eighty', \
#             90: 'Ninety', 0: 'Zero'}
# # unicode_line = unicode_line.translate(translation_table)
# data['App'] = data['App'].str.replace(' \d+ ', '')
# print(data['App'])
# tfidf = TfidfVectorizer(token_pattern='\\b[A-Za-z]+\\b', stop_words='english')
# tfidf_matrix = tfidf.fit_transform(data['App'])

# tfidf_matrix.shape
# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# cosine_sim[0]

# We now have a pairwise cosine similarity matrix for all the apps in our dataset. 
# The next step is to write a function that returns the 30 most similar apps based on the cosine similarity score.
data = data.reset_index()
indices = pd.Series(data.index, index=data['App'])

def get_recommendations(title, data, alg):
    scores = []
    sim = get_similarities(data, alg)
    print('sim score:')
    print(sim)
    print(sim.shape)
    if isinstance(indices[title], np.int64):
        sim_scores = list(enumerate(sim[indices[title]]))
        scores.extend(sim_scores)
    else:
        for idx in indices[title]:
            sim_scores = list(enumerate(sim[idx]))
            # # Get the top 100 results for each duplicate app <- found it doesnt have an effect.
            # sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # scores.extend(sim_scores[1:100])
            scores.extend(sim_scores)

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    scores = scores[1:31]
    
    app_indices = [i[0] for i in scores]
    new = data.iloc[app_indices]
    return ranked_recommendations(new)

#  THESE RECOMMENDATIONS ARE BASED ON GENRES AND RANKED
title = input("Please input an app title: ")
sim_alg = input("What algorithm would you like to score the similarities?\n - cosine\n - pearson\n - kendall\n - spearman\n")
# get_recommendations(title, data, sim_alg)
print(get_recommendations(title, data, sim_alg).head(40))