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
        self.train_data = None
        self.test_data = None
        self._build_model()

    def _build_model(self):
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            self.ratings[['userId', 'movieId', 'rating']], reader
        )
        self.train_data, self.test_data = train_test_split(data, test_size=0.2, random_state=42)
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

    def precision_recall_at_k(self, k=10, threshold=3.5):
        user_est_true = {}
        for uid, iid, true_r, est, _ in self.model.test(self.trainset.build_anti_testset()[:10000]):
            if uid not in user_est_true:
                user_est_true[uid] = []
            user_est_true[uid].append((est, true_r))

        precisions = {}
        recalls = {}
        for uid, user_ratings in user_est_true.items():
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
            n_rel_and_rec_k = sum(
                ((true_r >= threshold) and (est >= threshold))
                for (est, true_r) in user_ratings[:k]
            )
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        precision = sum(p for p in precisions.values()) / len(precisions)
        recall = sum(r for r in recalls.values()) / len(recalls)
        return {'precision@k': precision, 'recall@k': recall}

    def ndcg_at_k(self, k=10, threshold=3.5):
        user_est_true = {}
        for uid, iid, true_r, est, _ in self.model.test(self.trainset.build_anti_testset()[:10000]):
            if uid not in user_est_true:
                user_est_true[uid] = []
            user_est_true[uid].append((est, true_r))

        ndcgs = []
        for uid, user_ratings in user_est_true.items():
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            relevance = [1 if true_r >= threshold else 0 for (_, true_r) in user_ratings[:k]]

            dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
            ideal_relevance = sorted(relevance, reverse=True)
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))

            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)

        return {'ndcg@k': np.mean(ndcgs)}
