import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, movies_df=None, similarity_matrix=None):
        """
        Initialize the recommender with processed data or load from files.
        """
        if movies_df is not None and similarity_matrix is not None:
            self.movies_df = movies_df
            self.similarity_matrix = similarity_matrix
        else:
            self.load_data()
        
        # indices for faster lookups
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['title']).drop_duplicates()
    
    def load_data(self, data_dir='./data'):
        """
        Load processed data from pickle files.
        """
        try:
            self.movies_df = pd.read_pickle(os.path.join(data_dir, 'processed_movies.pkl'))
            with open(os.path.join(data_dir, 'similarity_matrix.pkl'), 'rb') as f:
                self.similarity_matrix = pickle.load(f)
        except FileNotFoundError:
            raise Exception("Processed data files not found. Run data_processing.py first.")
    
    def get_recommendations(self, title, n=10):
        """
        Get movie recommendations based on movie title.
        """
        if title not in self.indices:
            return []
        
        idx = self.indices[title]
        
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        sim_scores = sim_scores[1:n+1]
        
        movie_indices = [i[0] for i in sim_scores]
        
        recommendations = self.movies_df.iloc[movie_indices]
        
        return recommendations.to_dict('records')
    
    def search_movies(self, query, n=10):
        """
        Search for movies by title or genre.
        """
        query = query.lower()
        
        title_matches = self.movies_df[self.movies_df['title'].str.lower().str.contains(query, na=False)]
        
        genre_matches = pd.DataFrame()
        if isinstance(self.movies_df['genres'].iloc[0], list):
            genre_matches = self.movies_df[self.movies_df['genres'].apply(
                lambda x: any(query in genre.lower() for genre in x)
            )]
        else:
            genre_matches = self.movies_df[self.movies_df['genres'].str.lower().str.contains(query, na=False)]
        
        combined_results = pd.concat([title_matches, genre_matches])
        
        results = combined_results.drop_duplicates(subset=['movieId']).head(n)
        
        return results.to_dict('records')
    
    def get_movie_details(self, movie_id):
        """
        Get detailed information about a specific movie.
        """
        movie = self.movies_df[self.movies_df['movieId'] == movie_id]
        if movie.empty:
            return None
        
        return movie.iloc[0].to_dict()
