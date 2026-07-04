import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .preprocess import load_movies


class ContentBasedRecommender:
    def __init__(self):
        self.movies = load_movies()
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.cosine_sim = None
        self.indices = None
        self._build_model()

    def _build_model(self):
        self.movies['genres_str'] = self.movies['genres'].str.replace('|', ' ', regex=False)
        tfidf_matrix = self.tfidf.fit_transform(self.movies['genres_str'])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self.indices = pd.Series(
            self.movies.index, index=self.movies['title']
        ).drop_duplicates()

    def recommend(self, title, top_n=10):
        if title not in self.indices:
            return []

        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        results = self.movies.iloc[movie_indices][['movieId', 'title', 'genres']].copy()
        results['similarity'] = scores
        return results
