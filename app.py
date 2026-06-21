import os
import pickle
import html
import requests
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

# Set page configurations
st.set_page_config(
    page_title="Aura Cine Match - Intelligent Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject custom premium CSS styles
st.markdown(
    """
    <style>
    /* Custom Font (Outfit) */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* Global page styling */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Outfit', sans-serif !important;
        background: radial-gradient(circle at 10% 20%, rgb(10, 10, 18) 0%, rgb(18, 8, 30) 90%) !important;
        color: #e2e8f0 !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Custom Header Card */
    .main-header {
        text-align: center;
        padding: 40px 20px;
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 35px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
    }

    .main-header h1 {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa 0%, #ec4899 50%, #f43f5e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 10px 0;
        letter-spacing: -0.04em;
        line-height: 1.2;
    }

    .main-header p {
        font-size: 1.2rem;
        color: #94a3b8;
        font-weight: 300;
        margin: 0;
    }

    /* Search/Selection form container styling */
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        padding: 25px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 30px !important;
    }

    /* Streamlit Selectbox Label */
    div[data-testid="stForm"] label {
        color: #c084fc !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        margin-bottom: 8px !important;
    }

    /* Customize Selectbox input component */
    div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.04) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        transition: all 0.3s ease !important;
    }

    div[data-baseweb="select"]:hover {
        border-color: rgba(167, 139, 250, 0.5) !important;
    }

    div[data-baseweb="select"] > div {
        background-color: transparent !important;
        color: #ffffff !important;
    }

    /* Recommendations Title */
    .rec-section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 25px;
        margin-top: 15px;
        border-left: 4px solid #a78bfa;
        padding-left: 12px;
        letter-spacing: -0.02em;
    }

    /* Custom Streamlit Columns wrapper adjustment */
    [data-testid="column"] {
        padding: 0 10px !important;
    }

    /* Premium Movie Card Styling */
    .movie-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 14px;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
        display: flex;
        flex-direction: column;
        height: 570px; /* Rigid card height for visual alignment */
        position: relative;
    }

    .movie-card:hover {
        transform: translateY(-12px) scale(1.015);
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(167, 139, 250, 0.35);
        box-shadow: 0 20px 40px rgba(167, 139, 250, 0.12);
    }

    /* Movie Poster Image Container */
    .movie-poster-container {
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 12px;
        height: 320px; /* Fixed height for image uniform alignment */
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .movie-poster-container img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .movie-card:hover .movie-poster-container img {
        transform: scale(1.06);
    }

    /* Movie Details inside the card */
    .movie-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        height: 2.9rem; /* Keeps titles aligned even if 1 or 2 lines */
    }

    .movie-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }

    .rating-badge {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-flex;
        align-items: center;
        gap: 3px;
        border: 1px solid rgba(245, 158, 11, 0.15);
    }

    .year-badge {
        background: rgba(255, 255, 255, 0.05);
        color: #94a3b8;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Genre Tags */
    .genre-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        margin-bottom: 12px;
        height: 22px; /* Rigid row for genre tags alignment */
        overflow: hidden;
    }

    .genre-tag {
        background: rgba(167, 139, 250, 0.12);
        color: #c084fc;
        font-size: 0.72rem;
        padding: 2px 7px;
        border-radius: 9999px;
        font-weight: 600;
        border: 1px solid rgba(167, 139, 250, 0.1);
    }

    /* Movie overview/synopsis snippet */
    .movie-overview-section {
        font-size: 0.8rem;
        color: #94a3b8;
        line-height: 1.45;
        margin-top: auto; /* Push to bottom of card */
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        padding-top: 10px;
        display: -webkit-box;
        -webkit-line-clamp: 4;
        -webkit-box-orient: vertical;
        overflow: hidden;
        height: 5.8rem; /* Uniform overview height block */
    }

    /* Custom Streamlit Button Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #db2777 50%, #e11d48 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border-radius: 10px !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.35) !important;
        width: 100%;
        letter-spacing: 0.02em !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.55) !important;
        border-color: transparent !important;
    }

    div.stButton > button:active {
        transform: translateY(0px) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Render main page header
st.markdown(
    """
    <div class="main-header">
        <h1>AURA CINE MATCH</h1>
        <p>Intelligent Movie Recommendation System Powered by Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Function to fetch poster URL with caching
@st.cache_data(ttl=86400) # Cache poster URLs for 24 hours
def fetch_poster(movie_id):
    """Fetches the movie poster URL from TMDB API."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        response = requests.get(url, timeout=4)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except Exception:
        pass
    # Fallback to a placeholder if fetch fails
    return f"https://placehold.co/500x750/1e1b4b/cbd5e1?text=No+Poster"

# Recommendation logic using the optimized top-50 similarity matrix
def recommend(movie):
    """Recommends 5 similar movies based on the selected movie."""
    try:
        index = movies[movies['title'] == movie].index[0]
    except IndexError:
        st.error("Movie not found in the dataset. Please select another one.")
        return [], [], [], [], [], []
        
    # similarity[index] is a list of tuples (similar_index, score) precomputed and sorted
    distances = similarity[index]
    
    recommended_names = []
    recommended_posters = []
    recommended_years = []
    recommended_ratings = []
    recommended_overviews = []
    recommended_genres = []
    recommended_ids = []

    # Get details for the top 5 matches (excluding the input movie itself which is index 0)
    for i in distances[1:6]:
        idx = i[0]
        row = movies.iloc[idx]
        
        recommended_names.append(row.title)
        recommended_ids.append(row.movie_id)
        recommended_years.append(row.year)
        recommended_ratings.append(row.vote_average)
        recommended_overviews.append(row.get('original_overview', 'No description available.'))
        recommended_genres.append(row.get('original_genres', []))

    # Fetch posters in parallel using ThreadPoolExecutor for excellent loading speeds
    with ThreadPoolExecutor(max_workers=5) as executor:
        recommended_posters = list(executor.map(fetch_poster, recommended_ids))

    return (recommended_names, recommended_posters, recommended_years, 
            recommended_ratings, recommended_overviews, recommended_genres)

# Load data files
@st.cache_resource
def load_model_data():
    try:
        movies_dict = pickle.load(open('artifacts/movie_dict.pkl', 'rb'))
        movies_df = pd.DataFrame(movies_dict)
        similarity_matrix = pickle.load(open('artifacts/similarity.pkl', 'rb'))
        return movies_df, similarity_matrix
    except FileNotFoundError:
        return None, None

movies, similarity = load_model_data()

if movies is None or similarity is None:
    st.error("Model files not found in the artifacts directory. Please run the preprocessing script `process_data.py` first.")
    st.stop()

# Layout selection form
movie_list = movies['title'].values

with st.form("recommender_form"):
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown to discover match recommendations:",
        movie_list,
        index=0
    )
    submit_button = st.form_submit_button("Find Match Recommendations")

# Display recommendations
if submit_button:
    st.markdown('<div class="rec-section-title">Recommended Matches for you:</div>', unsafe_allow_html=True)
    
    with st.spinner('Calculating similarities and fetching movie details...'):
        (rec_names, rec_posters, rec_years, 
         rec_ratings, rec_overviews, rec_genres) = recommend(selected_movie)
         
    if rec_names:
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                # Escape html contents for security and template stability
                title_escaped = html.escape(rec_names[i])
                overview_escaped = html.escape(rec_overviews[i])
                poster_url = rec_posters[i]
                
                # Format Year
                year_val = rec_years[i]
                year_str = str(int(year_val)) if pd.notna(year_val) else "N/A"
                
                # Format Rating
                rating_val = rec_ratings[i]
                rating_str = f"{rating_val:.1f}" if pd.notna(rating_val) else "N/A"
                
                # Format Genres
                genres_list = rec_genres[i][:3]  # Show up to 3 genres
                genres_html = "".join([f'<span class="genre-tag">{html.escape(str(g))}</span>' for g in genres_list])
                
                # Render the premium movie card HTML
                card_html = f"""
                <div class="movie-card">
                    <div class="movie-poster-container">
                        <img src="{poster_url}" alt="{title_escaped}" loading="lazy">
                    </div>
                    <div class="movie-title">{title_escaped}</div>
                    <div class="movie-meta">
                        <span class="year-badge">📅 {year_str}</span>
                        <span class="rating-badge">⭐ {rating_str}</span>
                    </div>
                    <div class="genre-tags">
                        {genres_html}
                    </div>
                    <div class="movie-overview-section" title="{overview_escaped}">
                        {overview_escaped}
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
