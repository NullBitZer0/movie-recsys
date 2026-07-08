import streamlit as st
import pandas as pd
from src.recommend import MovieRecommender
from src.preprocess import load_movies, load_ratings
from src.tmdb import get_poster_url

st.set_page_config(page_title="MovieLens Recommender", page_icon="🎬", layout="wide")

@st.cache_resource
def load_recommender():
    return MovieRecommender()

def display_movie_card(title, genres, score_label, score_value, movie_id=None):
    col1, col2 = st.columns([1, 3])

    with col1:
        if movie_id:
            poster_url = get_poster_url(movie_id)
            if poster_url:
                st.image(poster_url, width=120)
            else:
                st.image("https://via.placeholder.com/120x180?text=No+Poster", width=120)
        else:
            st.image("https://via.placeholder.com/120x180?text=No+Poster", width=120)

    with col2:
        st.markdown(f"**{title}**")
        st.caption(f"Genres: {genres}")
        st.metric(score_label, f"{score_value:.2f}" if isinstance(score_value, float) else score_value)

def main():
    st.title("🎬 MovieLens Recommender System")
    st.markdown("---")

    recommender = load_recommender()
    movies = load_movies()
    ratings = load_ratings()

    tab1, tab2, tab3 = st.tabs(["Content-Based", "Collaborative", "Hybrid"])

    with tab1:
        st.header("Content-Based Recommendations")
        st.markdown("Find movies similar to a selected movie based on genre similarity.")

        movie_titles = sorted(movies['title'].unique())
        selected_movie = st.selectbox("Select a movie:", movie_titles, key="content_movie")

        if st.button("Get Recommendations", key="content_btn"):
            with st.spinner("Generating recommendations..."):
                recs = recommender.get_content_recommendations(selected_movie, top_n=10)

            st.subheader(f"Movies similar to: {selected_movie}")

            selected_movie_data = movies[movies['title'] == selected_movie]
            if not selected_movie_data.empty:
                selected_id = selected_movie_data.iloc[0]['movieId']
                poster_url = get_poster_url(selected_id)
                if poster_url:
                    st.image(poster_url, width=200)

            for _, row in recs.iterrows():
                display_movie_card(
                    row['title'],
                    row['genres'],
                    "Similarity",
                    row['similarity'],
                    row['movieId']
                )
                st.divider()

    with tab2:
        st.header("Collaborative Filtering Recommendations")
        st.markdown("Get personalized recommendations based on your rating history.")

        user_id = st.number_input(
            "Enter your User ID:",
            min_value=1,
            max_value=ratings['userId'].max(),
            value=1,
            key="collab_user"
        )

        if st.button("Get Recommendations", key="collab_btn"):
            with st.spinner("Generating personalized recommendations..."):
                recs = recommender.get_collaborative_recommendations(user_id, top_n=10)

            st.subheader(f"Recommended for User {user_id}")

            for _, row in recs.iterrows():
                display_movie_card(
                    row['title'],
                    row['genres'],
                    "Predicted Rating",
                    row['predicted_rating'],
                    row['movieId']
                )
                st.divider()

    with tab3:
        st.header("Hybrid Recommendations")
        st.markdown("Combine content-based and collaborative filtering for better results.")

        movie_titles = sorted(movies['title'].unique())
        selected_movie_hybrid = st.selectbox("Select a movie:", movie_titles, key="hybrid_movie")

        user_id_hybrid = st.number_input(
            "Enter your User ID (optional):",
            min_value=1,
            max_value=ratings['userId'].max(),
            value=1,
            key="hybrid_user"
        )

        if st.button("Get Recommendations", key="hybrid_btn"):
            with st.spinner("Generating hybrid recommendations..."):
                recs = recommender.get_hybrid_recommendations(
                    selected_movie_hybrid,
                    user_id=user_id_hybrid,
                    top_n=10
                )

            st.subheader(f"Hybrid Recommendations based on: {selected_movie_hybrid}")

            selected_movie_data = movies[movies['title'] == selected_movie_hybrid]
            if not selected_movie_data.empty:
                selected_id = selected_movie_data.iloc[0]['movieId']
                poster_url = get_poster_url(selected_id)
                if poster_url:
                    st.image(poster_url, width=200)

            for _, row in recs.iterrows():
                display_movie_card(
                    row['title'],
                    row['genres'],
                    "Score",
                    0,
                    row['movieId']
                )
                st.divider()

    st.markdown("---")
    st.sidebar.header("Dataset Stats")
    st.sidebar.metric("Total Movies", len(movies))
    st.sidebar.metric("Total Ratings", len(ratings))
    st.sidebar.metric("Total Users", ratings['userId'].nunique())

if __name__ == "__main__":
    main()
