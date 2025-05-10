import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import os
import time
import random
from requests.exceptions import ConnectionError
from recommender import MovieRecommender

st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Movie card styling */
    .movie-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
        height: 100%;
    }
    
    .movie-title {
        color: #ffffff;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    .movie-info {
        color: #cccccc;
        font-size: 14px;
    }
    
    /* Search box styling */
    .search-box {
        background-color: #2e2e2e;
        border-radius: 5px;
        padding: 20px;
        margin-bottom: 30px;
        margin-top: 0 !important;
    }
    
    /* Header styling */
    .header {
        background-color: #032541;
        padding: 15px 0;
        color: white;
        border-radius: 0;
        margin-bottom: 0 !important;
        width: 100%;
    }
    
    /* Search label styling */
    .search-label {
        font-size: 24px !important;
        font-weight: bold;
        margin-bottom: 15px;
        color: #ffffff;
    }
    
    /* Hide "Press Enter to apply" text */
    div[data-testid="InputInstructions"] {
        display: none;
    }
    
    /* Fix image display */
    .stImage img {
        display: block !important;
        max-width: 100% !important;
        height: auto !important;
        border-radius: 5px !important;
    }
    
    /* Remove any extra numbers */
    .stNumberInput {
        display: none !important;
    }
    
    /* Remove grey bars from sections */
    .css-1544g2n {
        padding-top: 0 !important;
    }
    
    /* Fix subheader styling */
    h3 {
        margin-top: 20px !important;
        margin-bottom: 20px !important;
        padding-top: 10px !important;
        border-top: none !important;
        font-size: 24px !important;
    }
    
    /* Remove default padding and margins */
    .stButton {
        margin-top: 10px;
    }
    
    /* Remove extra padding from columns */
    .row-widget.stHorizontal {
        gap: 10px;
        padding: 0 !important;
    }
    
    /* Remove all decorative horizontal lines */
    hr {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

try:
    TMDB_API_KEY = st.secrets["tmdb"]["api_key"]
except Exception:
    TMDB_API_KEY = "YOUR_TMDB_API_KEY"  # for local development

TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

@st.cache_resource
def load_recommender():
    """Load the movie recommender model or process data if not found."""
    try:
        return MovieRecommender()
    except Exception as e:
        
        from data_processing import load_and_process_data, compute_similarity_matrix, save_processed_data
        import os
        import requests
        import zipfile
        
        data_dir = './data'
        os.makedirs(data_dir, exist_ok=True)
        
        movielens_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        movielens_zip = os.path.join(data_dir, "ml-latest-small.zip")
        
        response = requests.get(movielens_url)
        with open(movielens_zip, 'wb') as f:
            f.write(response.content)
        
        with zipfile.ZipFile(movielens_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        movies_path = os.path.join(data_dir, "ml-latest-small", "movies.csv")
        links_path = os.path.join(data_dir, "ml-latest-small", "links.csv")
        ratings_path = os.path.join(data_dir, "ml-latest-small", "ratings.csv")
        
        movies_df = load_and_process_data(movies_path, links_path, ratings_path)
        similarity_matrix = compute_similarity_matrix(movies_df)
        save_processed_data(movies_df, similarity_matrix, data_dir)
    
        return MovieRecommender()


def fetch_poster(tmdb_id, max_retries=3):
    """Fetch movie poster from TMDB API with retry mechanism."""
    if not tmdb_id or not TMDB_API_KEY or TMDB_API_KEY == "YOUR_TMDB_API_KEY":
        return None
    
    for attempt in range(max_retries):
        try:
            url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                poster_path = data.get('poster_path')
                if poster_path:
                    poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path}"
                    img_response = requests.get(poster_url, timeout=5)
                    if img_response.status_code == 200:
                        return Image.open(BytesIO(img_response.content))
            return None
        except ConnectionError as e:
            if attempt < max_retries - 1:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Connection error, retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Error fetching poster: {e}")
                return None
        except Exception as e:
            print(f"Error fetching poster: {e}")
            return None
    
    return None

def display_movie_card(movie, col):
    """Display a movie card in the given column."""
    with col:
        st.markdown(f"<div class='movie-card'>", unsafe_allow_html=True)
        
        poster_img = None
        if 'tmdbId' in movie and movie['tmdbId'] and TMDB_API_KEY != "YOUR_TMDB_API_KEY":
            poster_img = fetch_poster(movie['tmdbId'])
        
        if poster_img:
            st.image(poster_img, use_container_width=True)
        else:
            clean_title = ''.join(c if c.isalnum() or c.isspace() else '+' for c in movie['title'].split('(')[0].strip())
            clean_title = clean_title.replace(' ', '+')
            st.image(f"https://via.placeholder.com/300x450/333333/FFFFFF?text={clean_title}", use_container_width=True)
        
        st.markdown(f"<div class='movie-title'>{movie['title']}</div>", unsafe_allow_html=True)
        
        if 'genres' in movie and movie['genres']:
            genres_text = ", ".join(movie['genres']) if isinstance(movie['genres'], list) else movie['genres']
            st.markdown(f"<div class='movie-info'>Genres: {genres_text}</div>", unsafe_allow_html=True)
        
        if 'rating' in movie and pd.notna(movie['rating']):
            st.markdown(f"<div class='movie-info'>Rating: ⭐ {movie['rating']:.1f}</div>", unsafe_allow_html=True)
        
        button_key = f"btn_{movie.get('movieId', hash(movie['title']))}"
        
        if st.button("Show Similar Movies", key=button_key):
            st.session_state.selected_movie = movie['title']
            st.session_state.show_recommendations = True
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

def main():
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'selected_movie' not in st.session_state:
        st.session_state.selected_movie = None
    if 'show_recommendations' not in st.session_state:
        st.session_state.show_recommendations = False
    
    recommender = load_recommender()
    if not recommender:
        st.stop()
    
    st.markdown("<div class='header'><h1 style='text-align: center; margin: 0;'>🎬 Movie Recommender System</h1></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='search-box'>", unsafe_allow_html=True)
    
    st.markdown("<p class='search-label'>Search for a movie by title or genre:</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        search_query = st.text_input(
            "Search Input",  
            value=st.session_state.search_query,
            key="search_input",
            label_visibility="collapsed"  
        )
    
    with col2:
        search_button = st.button("Search", key="search_button", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Real-time search suggestions
    if search_query and len(search_query) >= 3 and search_query != st.session_state.search_query:
        with st.spinner("Searching..."):
            st.session_state.search_results = recommender.search_movies(search_query, n=5)
            st.session_state.search_query = search_query
    
    
    if search_button and search_query:
        st.session_state.search_query = search_query
        st.session_state.search_results = recommender.search_movies(search_query, n=10)
        st.session_state.show_recommendations = False
        st.rerun()
    
    if st.session_state.show_recommendations and st.session_state.selected_movie:
        st.markdown(f"<h3>Movies similar to '{st.session_state.selected_movie}':</h3>", unsafe_allow_html=True)
        
        with st.spinner("Getting recommendations..."):
            recommendations = recommender.get_recommendations(st.session_state.selected_movie, n=10)
        
        if recommendations:
            cols = st.columns(5)
            for i, movie in enumerate(recommendations):
                display_movie_card(movie, cols[i % 5])
        else:
            st.info("No recommendations found for this movie.")
        
        if st.button("Back to Search Results"):
            st.session_state.show_recommendations = False
            st.rerun()
    
    elif st.session_state.search_results and not st.session_state.show_recommendations:
        st.markdown("<h3>Search Results:</h3>", unsafe_allow_html=True)
        cols = st.columns(5)
        for i, movie in enumerate(st.session_state.search_results):
            display_movie_card(movie, cols[i % 5])

if __name__ == "__main__":
    main()
