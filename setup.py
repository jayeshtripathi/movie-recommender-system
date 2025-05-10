import os
import requests
import zipfile
import pandas as pd
from data_processing import load_and_process_data, compute_similarity_matrix, save_processed_data

def download_file(url, destination):
    """Download a file from a URL to a destination path."""
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return
    
    print(f"Downloading {url} to {destination}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Download complete: {destination}")

def setup():
    """Set up the project by downloading and processing the dataset."""
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    movielens_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    movielens_zip = os.path.join(data_dir, "ml-latest-small.zip")
    
    download_file(movielens_url, movielens_zip)
    
    if os.path.exists(movielens_zip):
        with zipfile.ZipFile(movielens_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"Extracted {movielens_zip}")
    
    movies_path = os.path.join(data_dir, "ml-latest-small", "movies.csv")
    links_path = os.path.join(data_dir, "ml-latest-small", "links.csv")
    ratings_path = os.path.join(data_dir, "ml-latest-small", "ratings.csv")
    
    if os.path.exists(movies_path):
        print("Processing MovieLens dataset...")
        
        movies_df = load_and_process_data(movies_path, links_path, ratings_path)
        
        similarity_matrix = compute_similarity_matrix(movies_df)
        
        save_processed_data(movies_df, similarity_matrix, data_dir)
        
        print(f"Processed {len(movies_df)} movies from MovieLens dataset.")
    else:
        print("MovieLens dataset files not found.")
    
    print("Setup complete!")

if __name__ == "__main__":
    setup()
