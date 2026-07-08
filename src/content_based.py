import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .preprocess import load_movies, load_ratings


class ContentBasedRecommender:
    def __init__(self):
        self.movies = load_movies()
        self.ratings = load_ratings()
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

    def genre_consistency(self, title, top_n=10):
        recs = self.recommend(title, top_n)
        if len(recs) == 0:
            return 0

        source_movie = self.movies[self.movies['title'] == title].iloc[0]
        source_genres = set(source_movie['genres'].split('|'))

        consistency_scores = []
        for _, row in recs.iterrows():
            rec_genres = set(row['genres'].split('|'))
            overlap = len(source_genres.intersection(rec_genres))
            total = len(source_genres.union(rec_genres))
            consistency_scores.append(overlap / total if total > 0 else 0)

        return np.mean(consistency_scores)

    def precision_recall_at_k(self, user_id, k=10, threshold=3.5):
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        relevant_movies = set(
            user_ratings[user_ratings['rating'] >= threshold]['movieId']
        )

        if not relevant_movies:
            return {'precision@k': 0, 'recall@k': 0}

        user_high_rated = user_ratings[user_ratings['rating'] >= threshold]
        if len(user_high_rated) == 0:
            return {'precision@k': 0, 'recall@k': 0}

        sample_movie_id = user_high_rated.iloc[0]['movieId']
        sample_movie = self.movies[self.movies['movieId'] == sample_movie_id]

        if sample_movie.empty:
            return {'precision@k': 0, 'recall@k': 0}

        sample_title = sample_movie.iloc[0]['title']
        recs = self.recommend(sample_title, top_n=k)

        if len(recs) == 0:
            return {'precision@k': 0, 'recall@k': 0}

        recommended_ids = set(recs['movieId'].tolist())

        relevant_and_recommended = len(relevant_movies.intersection(recommended_ids))
        precision = relevant_and_recommended / k
        recall = relevant_and_recommended / len(relevant_movies)

        return {'precision@k': precision, 'recall@k': recall}

    def evaluate(self, sample_size=50, k=10):
        user_ids = self.ratings['userId'].unique()
        sample_users = np.random.choice(user_ids, min(sample_size, len(user_ids)), replace=False)

        precisions = []
        recalls = []
        consistencies = []

        for user_id in sample_users:
            user_ratings = self.ratings[self.ratings['userId'] == user_id]
            high_rated = user_ratings[user_ratings['rating'] >= 3.5]

            if len(high_rated) == 0:
                continue

            sample_movie_id = high_rated.iloc[0]['movieId']
            sample_movie = self.movies[self.movies['movieId'] == sample_movie_id]

            if sample_movie.empty:
                continue

            sample_title = sample_movie.iloc[0]['title']
            pr = self.precision_recall_at_k(user_id, k=k)
            consistency = self.genre_consistency(sample_title, top_n=k)

            precisions.append(pr['precision@k'])
            recalls.append(pr['recall@k'])
            consistencies.append(consistency)

        return {
            'precision@k': np.mean(precisions) if precisions else 0,
            'recall@k': np.mean(recalls) if recalls else 0,
            'genre_consistency': np.mean(consistencies) if consistencies else 0
        }
