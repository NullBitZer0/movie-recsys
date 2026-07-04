from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender
from .preprocess import load_movies, load_ratings


class MovieRecommender:
    def __init__(self):
        self.content_recommender = ContentBasedRecommender()
        self.collab_recommender = CollaborativeRecommender()
        self.movies = load_movies()
        self.ratings = load_ratings()

    def get_content_recommendations(self, title, top_n=10):
        return self.content_recommender.recommend(title, top_n)

    def get_collaborative_recommendations(self, user_id, top_n=10):
        return self.collab_recommender.recommend_for_user(user_id, top_n)

    def get_hybrid_recommendations(self, title, user_id=None, top_n=10):
        content_recs = self.content_recommender.recommend(title, top_n=top_n * 2)

        if user_id is not None:
            collab_recs = self.collab_recommender.recommend_for_user(user_id, top_n=top_n * 2)
            content_ids = set(content_recs['movieId'].tolist())
            collab_ids = set(collab_recs['movieId'].tolist())

            hybrid_ids = list(content_ids.union(collab_ids))[:top_n]
            results = self.movies[self.movies['movieId'].isin(hybrid_ids)].copy()
            return results

        return content_recs.head(top_n)

    def get_movie_details(self, movie_id):
        movie = self.movies[self.movies['movieId'] == movie_id]
        if movie.empty:
            return None
        return movie.iloc[0]

    def get_user_stats(self, user_id):
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        return {
            'total_ratings': len(user_ratings),
            'avg_rating': user_ratings['rating'].mean(),
            'rating_distribution': user_ratings['rating'].value_counts().sort_index().to_dict()
        }
