import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv('google-play-store-apps/googleplaystore.csv')
#wait like 30s for this to finish
# to_drop = ['Reviews',
#           'Size',
#           'Genres',
#           'Last Updated',
#           'Current Ver',
#           'Android Ver']
# data.drop(to_drop, inplace=True, axis=1)

# installs = r'^(\d*)'
# extr = data["Installs"].str.replace(",", '')
# extr = extr.str.extract(installs)
# extr.head()

# Fixing Price
data = data.where(data['Price'] != "Everyone")
data["Price"] = data["Price"].str.replace("$", '')
data["Price"] = pd.to_numeric(data["Price"])
# data["Price"] = data.apply(lambda x: pd.to_numeric(x), axis=0)

# Fixing Installs
data["Installs"] = data["Installs"].str.replace(",", '')
data["Installs"] = data["Installs"].str.replace("+", '')
data["Installs"] = pd.to_numeric(data["Installs"])
# data["Installs"] = extr.apply(lambda x: pd.to_numeric(x), axis=0)
# data["Reviews"] = extr.apply(lambda x: pd.to_numeric(x), axis=0)

data["Reviews"] = pd.to_numeric(data["Reviews"])


C = data['Rating'].mean()
print(C)

# Calculate the minimum number of votes required to be in the chart, m
m = data['Reviews'].quantile(0.90)
print(m)

# Filter out all qualified apps into a new DataFrame
q_apps = data.copy().loc[data['Reviews'] >= m]
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

#Print the top 5 apps
# print(q_apps)


#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
data['Category'] = data['Category'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(data['Category'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construct a reverse map of indices and app titles
indices = pd.Series(data.index, index=data['App']).drop_duplicates()


# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(data['Category'])
# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
# Reset index of your main DataFrame and construct reverse mapping as before
data = data.reset_index()
indices = pd.Series(data.index, index=data['App'])


def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the app that matches the name
    idx = indices[title]

    # Get the pairwsie similarity scores of all apps with that app
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the apps based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[0], reverse=True)

    # Get the scores of the 10 most similar apps
    sim_scores = sim_scores[1:11]

    # Get the app indices
    app_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar app
    return data['App'].iloc[app_indices]

print(get_recommendations('Paint Space AR', cosine_sim2))