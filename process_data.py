import os
import ast
import pickle
import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def convert(text):
    L = []
    try:
        for i in ast.literal_eval(text):
            L.append(i['name'])
    except (ValueError, SyntaxError):
        pass
    return L

def convert_cast(text):
    L = []
    counter = 0
    try:
        for i in ast.literal_eval(text):
            if counter < 3:
                L.append(i['name'])
            counter += 1
    except (ValueError, SyntaxError):
        pass
    return L

def fetch_director(text):
    L = []
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
    except (ValueError, SyntaxError):
        pass
    return L

def remove_space(L):
    return [i.replace(" ", "") for i in L]

def main():
    print("Loading datasets...")
    movies_path = os.path.join("data", "tmdb_5000_movies.csv")
    credits_path = os.path.join("data", "tmdb_5000_credits.csv")
    
    if not os.path.exists(movies_path) or not os.path.exists(credits_path):
        print("Error: Dataset files not found in data/ directory. Please run download_dataset.py first.")
        return
        
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    
    print("Merging datasets...")
    movies = movies.merge(credits, on='title')
    
    # Select columns to match the Indian movies schema
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'release_date', 'vote_average']]
    
    # Import and concatenate Indian movies
    from indian_movies import INDIAN_MOVIES
    indian_df = pd.DataFrame(INDIAN_MOVIES)
    print("Integrating Hindi & South Indian movies dataset...")
    movies = pd.concat([movies, indian_df], ignore_index=True)
    
    # Keep release date and handle year before dropping NaNs or during cleanup
    movies.dropna(subset=['release_date', 'overview', 'title'], inplace=True)
    movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year
    
    print("Preprocessing columns...")
    # Keep original genres and overview for UI display
    movies['original_genres'] = movies['genres'].apply(convert)
    movies['original_overview'] = movies['overview']
    
    # Process attributes for similarity matching
    movies['genres'] = movies['genres'].apply(convert).apply(remove_space)
    movies['keywords'] = movies['keywords'].apply(convert).apply(remove_space)
    movies['cast'] = movies['cast'].apply(convert_cast).apply(remove_space)
    movies['crew'] = movies['crew'].apply(fetch_director).apply(remove_space)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    
    # Create tags
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    # Drop intermediate columns and keep final ones
    new_df = movies[['movie_id', 'title', 'tags', 'year', 'vote_average', 'original_overview', 'original_genres']]
    
    # Convert tags back to string and lowercase
    new_df = new_df.copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x)).apply(lambda x: x.lower())
    
    print("Stemming tags...")
    ps = PorterStemmer()
    def stems(text):
        T = [ps.stem(word) for word in text.split()]
        return " ".join(T)
        
    new_df['tags'] = new_df['tags'].apply(stems)
    
    print("Vectorizing tags...")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(new_df['tags']).toarray()
    
    print("Calculating cosine similarity matrix...")
    similarity = cosine_similarity(vector)
    
    # Create artifacts directory if not exists
    os.makedirs("artifacts", exist_ok=True)
    
    print("Optimizing similarity matrix...")
    # Instead of storing 4805x4805 floats (184MB), pre-sort and store top 50 recommendations per movie.
    # This reduces size to ~1.5MB and makes recommendations instantaneous.
    optimized_similarity = []
    for i in range(len(similarity)):
        # Get sorted list of (index, similarity_score)
        distances = sorted(list(enumerate(similarity[i])), reverse=True, key=lambda x: x[1])
        # Save only the top 50 matches (including the movie itself at index 0)
        optimized_similarity.append(distances[:50])
        
    print("Saving processed data and optimized model to artifacts/...")
    
    # Save the dictionary of movie data
    with open(os.path.join("artifacts", "movie_dict.pkl"), "wb") as f:
        pickle.dump(new_df.to_dict(), f)
        
    # Save the optimized similarity list
    with open(os.path.join("artifacts", "similarity.pkl"), "wb") as f:
        pickle.dump(optimized_similarity, f)
        
    print("Data processing and model generation completed successfully!")

if __name__ == "__main__":
    main()
