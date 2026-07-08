import pandas as pd
import numpy as np
from .preprocess import load_movies, load_ratings


class PopularityRecommender:
    def __init__(self):
        self.movies = load_movies()
        self.ratings = load_ratings()
        self.movie_stats = self._compute_stats()

    def _compute_stats(self):
        stats = self.ratings.groupby('movieId').agg(
            avg_rating=('rating', 'mean'),
            rating_count=('rating', 'count')
        ).reset_index()

        min_votes = stats['rating_count'].quantile(0.9)
        qualified = stats[stats['rating_count'] >= min_votes]

        C = qualified['avg_rating'].mean()
        m = min_votes

        qualified['weighted_score'] = (
            (qualified['rating_count'] / (qualified['rating_count'] + m)) * qualified['avg_rating'] +
            (m / (qualified['rating_count'] + m)) * C
        )

        return qualified.merge(self.movies[['movieId', 'title', 'genres']], on='movieId')

    def recommend(self, top_n=10):
        return self.movie_stats.nlargest(top_n, 'weighted_score')[
            ['movieId', 'title', 'genres', 'avg_rating', 'rating_count', 'weighted_score']
        ]

    def recommend_for_user(self, user_id, top_n=10):
        user_rated = self.ratings[self.ratings['userId'] == user_id]['movieId'].tolist()
        recs = self.movie_stats[~self.movie_stats['movieId'].isin(user_rated)]
        return recs.nlargest(top_n, 'weighted_score')[
            ['movieId', 'title', 'genres', 'avg_rating', 'rating_count', 'weighted_score']
        ]

    def precision_recall_at_k(self, user_id, k=10, threshold=3.5):
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        relevant_movies = set(
            user_ratings[user_ratings['rating'] >= threshold]['movieId']
        )

        if not relevant_movies:
            return {'precision@k': 0, 'recall@k': 0}

        recs = self.recommend_for_user(user_id, top_n=k)
        recommended_ids = set(recs['movieId'].tolist())

        relevant_and_recommended = len(relevant_movies.intersection(recommended_ids))
        precision = relevant_and_recommended / k
        recall = relevant_and_recommended / len(relevant_movies)

        return {'precision@k': precision, 'recall@k': recall}
