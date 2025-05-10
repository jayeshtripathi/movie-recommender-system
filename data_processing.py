import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle
import os

def load_and_process_data(movies_path, links_path=None, ratings_path=None):
    """
    Load and process movie data from MovieLens dataset.
    """
    movies_df = pd.read_csv(movies_path)
    
    movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|') if x != '(no genres listed)' else [])
    
    if links_path and os.path.exists(links_path):
        links_df = pd.read_csv(links_path)
        movies_df = movies_df.merge(links_df, on='movieId', how='left')
    
    if ratings_path and os.path.exists(ratings_path):
        ratings_df = pd.read_csv(ratings_path)
        avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
        movies_df = movies_df.merge(avg_ratings, on='movieId', how='left')
    
    movies_df['combined_features'] = movies_df.apply(
        lambda row: f"{row['title']} {' '.join(row['genres'])}", 
        axis=1
    )
    
    return movies_df

def compute_similarity_matrix(df):
    """
    Compute cosine similarity matrix based on TF-IDF vectors.
    """
    # Create TF-IDF vectors
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim

def save_processed_data(df, similarity_matrix, output_dir='./data'):
    """
    Save processed dataframe and similarity matrix.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df.to_pickle(os.path.join(output_dir, 'processed_movies.pkl'))
    with open(os.path.join(output_dir, 'similarity_matrix.pkl'), 'wb') as f:
        pickle.dump(similarity_matrix, f)

def main():
    data_dir = './data/ml-latest-small'
    movies_path = os.path.join(data_dir, 'movies.csv')
    links_path = os.path.join(data_dir, 'links.csv')
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    
    movies_df = load_and_process_data(movies_path, links_path, ratings_path)
    
    similarity_matrix = compute_similarity_matrix(movies_df)
    
    save_processed_data(movies_df, similarity_matrix)
    
    print(f"Processed {len(movies_df)} movies and saved data.")

if __name__ == "__main__":
    main()
