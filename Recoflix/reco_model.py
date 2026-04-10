import pandas as pd
import numpy as np
import pickle
import ast

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge
movies = movies.merge(credits, on='title')


# ---------------------------------------------------
# SELECT FEATURES
# ---------------------------------------------------
movies = movies[['title', 'genres', 'keywords', 'cast', 'crew', 'vote_average']]


# ---------------------------------------------------
# FUNCTIONS TO CLEAN JSON
# ---------------------------------------------------
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L


def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
    return L


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L


# Apply
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)


# ---------------------------------------------------
# CREATE TAGS
# ---------------------------------------------------
movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))


# ---------------------------------------------------
# VECTORIZE
# ---------------------------------------------------
cv = CountVectorizer(max_features=8000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)


# ---------------------------------------------------
# ML PART (CLASSIFICATION)
# ---------------------------------------------------

# Label
movies['label'] = movies['vote_average'].apply(lambda x: 1 if x >= 7 else 0)

X = vectors
y = movies['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 Better model
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------------------------------------------------
# PRINT METRICS
# ---------------------------------------------------
print("\n🔥 MODEL EVALUATION 🔥")

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ---------------------------------------------------
# SAVE FILES
# ---------------------------------------------------
pickle.dump(movies, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("\n✅ Model & similarity saved successfully!")