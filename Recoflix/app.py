from flask import Flask, render_template, request
import pickle
import pandas as pd
import random

app = Flask(__name__)

# Load data
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))


# ---------------------------------------------------
# 🔍 SEARCH RECOMMENDATION (WITH SELECTED MOVIE)
# ---------------------------------------------------
def recommend(movie):
    movie_lower = movie.lower()

    if movie_lower not in movies['title'].str.lower().values:
        return None, [], []

    index = movies[movies['title'].str.lower() == movie_lower].index[0]

    # 🎯 Selected movie
    selected_movie = movies.iloc[index]['title']
    selected_rating = round(movies.iloc[index]['vote_average'], 1)

    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:10]

    rec = []
    ratings = []

    for i in movie_list:
        rec.append(movies.iloc[i[0]]['title'])
        ratings.append(round(movies.iloc[i[0]]['vote_average'], 1))

    return (selected_movie, selected_rating), rec, ratings


# ---------------------------------------------------
# 🎭 CATEGORY RECOMMENDATION
# ---------------------------------------------------
def recommend_by_category(g1, g2, g3):
    filtered = movies[
        movies['genres'].apply(lambda x: g1 in x or g2 in x or g3 in x)
    ]

    if len(filtered) < 20:
        sample = filtered
    else:
        sample = filtered.sample(20)

    rec = list(sample['title'])
    ratings = list(sample['vote_average'].round(1))

    return rec, ratings


# ---------------------------------------------------
# HOME
# ---------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')


# ---------------------------------------------------
# CATEGORY ROUTE
# ---------------------------------------------------
@app.route('/suggest', methods=['POST'])
def suggest():
    try:
        g1 = request.form['genre1']
        g2 = request.form['genre2']
        g3 = request.form['genre3']

        rec, ratings = recommend_by_category(g1, g2, g3)

        top5 = list(zip(rec[:5], ratings[:5]))
        others = list(zip(rec[5:], ratings[5:]))

        return render_template(
            'index.html',
            top5=top5,
            others=others
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=f"Error: {str(e)}"
        )


# ---------------------------------------------------
# SEARCH ROUTE
# ---------------------------------------------------
@app.route('/recommend', methods=['POST'])
def recommend_movies():
    try:
        movie = request.form['movie']

        selected, rec, ratings = recommend(movie)

        if selected is None:
            return render_template(
                'index.html',
                error="Movie not found 😢 Try another one"
            )

        top3 = list(zip(rec[:3], ratings[:3]))
        others = list(zip(rec[3:], ratings[3:]))

        return render_template(
            'index.html',
            selected=selected,
            top3=top3,
            others=others
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=f"Error: {str(e)}"
        )


# ---------------------------------------------------
# RUN
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)