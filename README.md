# Movie Recommender System

A content-based movie recommendation system built with Streamlit that allows users to search for movies by title or genre and get personalized recommendations.

## Live Demo

[https://mov-recommender.streamlit.app/](https://mov-recommender.streamlit.app/)



## How It Works

This movie recommender system uses content-based filtering to suggest similar movies:

1. **Data Processing**: The system processes movie data from the MovieLens dataset, including titles, genres, and user ratings.
2. **Feature Extraction**: Movie features (genres, titles) are converted into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
3. **Similarity Calculation**: Cosine similarity is computed between all movie vectors to create a similarity matrix.
4. **Recommendation Generation**: When a user selects a movie, the system finds the most similar movies based on the pre-computed similarity matrix.
5. **Visual Presentation**: Results are presented with movie posters fetched from The Movie Database (TMDB) API.



## Local Development

To run this project locally:

1. Clone the repository:

```
git clone https://github.com/jayeshtripathi/movie-recommender-system.git
cd movie-recommender-system
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Get a TMDB API key:
    - Sign up at [The Movie Database](https://www.themoviedb.org/signup)
    - Go to Settings > API and request an API key
4. Run the app:

```
streamlit run app.py
```


