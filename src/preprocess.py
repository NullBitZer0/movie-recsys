import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_movies():
    return pd.read_csv(os.path.join(DATA_DIR, 'movies.csv'))


def load_ratings():
    return pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))


def load_tags():
    return pd.read_csv(os.path.join(DATA_DIR, 'tags.csv'))


def load_links():
    return pd.read_csv(os.path.join(DATA_DIR, 'links.csv'))


def get_movie_titles():
    movies = load_movies()
    return dict(zip(movies['movieId'], movies['title']))


def get_genre_matrix():
    movies = load_movies()
    genres = movies['genres'].str.get_dummies(sep='|')
    genres = genres.drop(columns=['(no genres listed)'], errors='ignore')
    return movies[['movieId', 'title']], genres
