import streamlit as st
import pandas as pd
from src.recommend import MovieRecommender
from src.preprocess import load_movies, load_ratings

st.set_page_config(page_title="MovieLens Recommender", page_icon="🎬", layout="wide")

@st.cache_resource
def load_recommender():
    return MovieRecommender()

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
            for _, row in recs.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{row['title']}**")
                    st.caption(f"Genres: {row['genres']}")
                with col2:
                    st.metric("Similarity", f"{row['similarity']:.2f}")

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
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{row['title']}**")
                    st.caption(f"Genres: {row['genres']}")
                with col2:
                    st.metric("Predicted Rating", f"{row['predicted_rating']:.2f}")

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
            for _, row in recs.iterrows():
                st.write(f"**{row['title']}** - {row['genres']}")

    st.markdown("---")
    st.sidebar.header("Dataset Stats")
    st.sidebar.metric("Total Movies", len(movies))
    st.sidebar.metric("Total Ratings", len(ratings))
    st.sidebar.metric("Total Users", ratings['userId'].nunique())

if __name__ == "__main__":
    main()
