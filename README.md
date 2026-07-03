# MovieLens Recommender System

A movie recommendation system built using the MovieLens 100K dataset.

## Features

- **Content-Based Filtering**: Recommends movies based on genre similarity using TF-IDF and cosine similarity
- **Collaborative Filtering**: Personalized recommendations using SVD (Singular Value Decomposition)
- **Hybrid Approach**: Combines both methods for better recommendations
- **Interactive UI**: Streamlit web application for easy interaction

## Project Structure

```
movie-lens/
├── data/                  # MovieLens dataset files
├── notebooks/
│   └── eda.ipynb         # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── preprocess.py     # Data loading and preprocessing
│   ├── content_based.py  # Content-based recommender
│   ├── collaborative.py  # Collaborative filtering
│   └── recommend.py      # Hybrid recommendation engine
├── app.py                # Streamlit web application
├── requirements.txt      # Python dependencies
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the Web App

```bash
streamlit run app.py
```

### Run EDA Notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

## How It Works

### Content-Based Filtering
- Uses TF-IDF vectorization on movie genres
- Computes cosine similarity between movies
- Recommends movies with highest similarity scores

### Collaborative Filtering
- Uses SVD algorithm from the Surprise library
- Learns latent factors from user-item interactions
- Predicts ratings for unseen movies

### Hybrid Approach
- Combines content-based and collaborative recommendations
- Provides diverse and personalized suggestions

## Dataset

MovieLens 100K dataset containing:
- 100,836 ratings
- 9,742 movies
- 610 users
- Rating scale: 0.5 - 5.0 (half-star increments)

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Surprise (for SVD)
- Streamlit
- Matplotlib, Seaborn
