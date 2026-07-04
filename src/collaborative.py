import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from .preprocess import load_ratings, load_movies


class CollaborativeRecommender:
    def __init__(self):
        self.model = SVD(n_factors=100, n_epochs=20, random_state=42)
        self.ratings = load_ratings()
        self.movies = load_movies()
        self.trainset = None
        self.testset = None
        self._build_model()

    def _build_model(self):
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            self.ratings[['userId', 'movieId', 'rating']], reader
        )
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)

    def predict(self, user_id, movie_id):
        pred = self.model.predict(user_id, movie_id)
        return pred.est

    def recommend_for_user(self, user_id, top_n=10):
        user_ratings = self.ratings[self.ratings['userId'] == user_id]['movieId'].tolist()
        all_movies = self.movies['movieId'].unique()
        unrated = [m for m in all_movies if m not in user_ratings]

        predictions = []
        for movie_id in unrated:
            pred = self.predict(user_id, movie_id)
            predictions.append((movie_id, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:top_n]

        movie_ids = [p[0] for p in top_predictions]
        scores = [p[1] for p in top_predictions]

        results = self.movies[self.movies['movieId'].isin(movie_ids)].copy()
        results['predicted_rating'] = results['movieId'].map(dict(zip(movie_ids, scores)))
        return results.sort_values('predicted_rating', ascending=False)

    def evaluate(self):
        testset = self.trainset.build_anti_testset()
        predictions = self.model.test(testset[:10000])
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        return {'rmse': rmse, 'mae': mae}
