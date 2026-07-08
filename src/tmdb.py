import requests
import pandas as pd
import os
from functools import lru_cache

TMDB_API_KEY = os.getenv('TMDB_API_KEY', '28a1f0a0ab25ce2e78cb900cb3f0dfb1')


def load_links():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'links.csv'))


@lru_cache(maxsize=1000)
def get_poster_url(movie_id):
    links = load_links()
    movie_link = links[links['movieId'] == movie_id]

    if movie_link.empty:
        return None

    tmdb_id = movie_link.iloc[0]['tmdbId']

    if pd.isna(tmdb_id):
        return None

    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass

    return None


@lru_cache(maxsize=1000)
def get_movie_details(movie_id):
    links = load_links()
    movie_link = links[links['movieId'] == movie_id]

    if movie_link.empty:
        return None

    tmdb_id = movie_link.iloc[0]['tmdbId']

    if pd.isna(tmdb_id):
        return None

    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}&append_to_response=credits"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass

    return None


def search_movie(title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                return results[0]
    except Exception:
        pass

    return None
