import os
import pickle
import pandas as pd
import numpy as np
import db_helper

class HybridRecommender:
    def __init__(self):
        self.movies = None
        self.similarity = None
        self.load_models()

    def load_models(self):
        try:
            movies_dict = pickle.load(open('artifacts/movie_dict.pkl', 'rb'))
            self.movies = pd.DataFrame(movies_dict)
            self.similarity = pickle.load(open('artifacts/similarity.pkl', 'rb'))
        except FileNotFoundError:
            self.movies = None
            self.similarity = None

    def get_movie_index(self, title):
        try:
            return self.movies[self.movies['title'] == title].index[0]
        except (IndexError, AttributeError):
            return None

    def get_movie_by_id(self, movie_id):
        if self.movies is None:
            return None
        row = self.movies[self.movies['movie_id'] == movie_id]
        if not row.empty:
            return row.iloc[0]
        return None

    def get_movie_index_by_id(self, movie_id):
        try:
            return self.movies[self.movies['movie_id'] == movie_id].index[0]
        except (IndexError, AttributeError):
            return None

    def get_hybrid_recommendations(self, user_id, limit=5):
        """
        Calculates hybrid recommendations based on content similarity and user likes/ratings.
        Weighted by user ratings (1-5).
        """
        if self.movies is None or self.similarity is None:
            return []

        # Get user likes from database
        user_likes = db_helper.get_likes(user_id)
        if not user_likes:
            # If no likes, return top trending movies (sorted by vote_average)
            trending = self.movies.sort_values(by='vote_average', ascending=False)
            return [
                {
                    "movie_id": row.movie_id,
                    "title": row.title,
                    "year": row.year,
                    "vote_average": row.vote_average,
                    "overview": row.get('original_overview', ''),
                    "genres": row.get('original_genres', []),
                    "explanation": "Trending choice (new user)"
                }
                for _, row in trending.head(limit).iterrows()
            ]

        # Accumulator for similar movie indices: index -> weighted_score
        scores = {}
        # Track movies the user already liked to exclude them
        liked_indices = set()
        
        # Details of user likes to construct explanations: movie_id -> {title, rating, index}
        liked_details = {}

        for like in user_likes:
            m_id = like['movie_id']
            m_rating = like['rating']
            m_title = like['title']
            
            idx = self.get_movie_index_by_id(m_id)
            if idx is not None:
                liked_indices.add(idx)
                liked_details[idx] = {
                    "title": m_title,
                    "rating": m_rating,
                    "movie_id": m_id
                }

        # Calculate recommendation weights
        for liked_idx, details in liked_details.items():
            rating_weight = details['rating'] / 5.0 # Scale rating to 0.2 - 1.0
            
            # similarity[liked_idx] is a list of tuples (similar_idx, similarity_score)
            similar_movies = self.similarity[liked_idx]
            
            for sim_idx, sim_score in similar_movies:
                if sim_idx in liked_indices:
                    continue # Skip already liked/watched movies
                
                # Add to weighted score
                weighted_score = sim_score * rating_weight
                if sim_idx not in scores:
                    scores[sim_idx] = {
                        "total_score": 0.0,
                        "contributions": []
                    }
                
                scores[sim_idx]["total_score"] += weighted_score
                scores[sim_idx]["contributions"].append({
                    "liked_idx": liked_idx,
                    "liked_title": details['title'],
                    "score": sim_score
                })

        # Sort recommendations by total score descending
        sorted_recommendations = sorted(scores.items(), key=lambda x: x[1]["total_score"], reverse=True)

        recommendations_list = []
        for sim_idx, score_info in sorted_recommendations[:limit]:
            row = self.movies.iloc[sim_idx]
            
            # Find the primary contributor for explainable AI reasoning
            # We sort contributions by similarity score descending
            contributions = sorted(score_info["contributions"], key=lambda x: x["score"], reverse=True)
            primary = contributions[0]
            
            # Simple formatting of percentage similarity
            similarity_pct = int(primary["score"] * 100)
            explanation = f"Because you liked **{primary['liked_title']}** ({similarity_pct}% similarity)"
            
            if len(contributions) > 1:
                secondary = contributions[1]
                sec_pct = int(secondary["score"] * 100)
                explanation += f" and also matches **{secondary['liked_title']}** ({sec_pct}%)"

            recommendations_list.append({
                "movie_id": row.movie_id,
                "title": row.title,
                "year": row.year,
                "vote_average": row.vote_average,
                "overview": row.get('original_overview', ''),
                "genres": row.get('original_genres', []),
                "explanation": explanation
            })

        # If we didn't get enough recommendations, backfill with trending movies not already in list
        if len(recommendations_list) < limit:
            existing_ids = {r["movie_id"] for r in recommendations_list}
            existing_ids.update({like["movie_id"] for like in user_likes})
            
            trending = self.movies.sort_values(by='vote_average', ascending=False)
            for _, row in trending.iterrows():
                if len(recommendations_list) >= limit:
                    break
                if row.movie_id not in existing_ids:
                    recommendations_list.append({
                        "movie_id": row.movie_id,
                        "title": row.title,
                        "year": row.year,
                        "vote_average": row.vote_average,
                        "overview": row.get('original_overview', ''),
                        "genres": row.get('original_genres', []),
                        "explanation": "Recommended trending hit"
                    })

        return recommendations_list

    def get_natural_language_recommendations(self, query, limit=5):
        """
        Parses a natural language query (e.g. 'Suggest me a sad romantic movie like Titanic')
        extracts intent using keywords and tags matching, and returns movies.
        """
        if self.movies is None:
            return []

        query = query.lower()
        
        # Step 1: Scan for movie titles in the query
        referenced_index = None
        for idx, row in self.movies.iterrows():
            title = row.title.lower()
            if len(title) > 3 and title in query:
                referenced_index = idx
                break

        # Step 2: Look for genre/mood keywords in query
        mood_keywords = {
            "romantic": ["romanc", "love", "heart", "sad romantic"],
            "sad": ["sad", "emotional", "cry", "tear", "depressed"],
            "happy": ["happy", "funny", "laugh", "feel-good", "comedy"],
            "thriller": ["thriller", "suspense", "scary", "mystery", "murder"],
            "action": ["action", "fight", "blast", "war", "explos"],
            "sci-fi": ["sci-fi", "science fiction", "space", "alien", "future"],
            "fantasy": ["fantasy", "magic", "sword", "dragon"]
        }

        active_keywords = []
        for key, synonyms in mood_keywords.items():
            for syn in synonyms:
                if syn in query:
                    active_keywords.append(key)
                    break

        # Step 3: Rank movies
        # Base rank: start with all zeros
        movie_ranks = np.zeros(len(self.movies))

        # Weight 1: Similarity to referenced movie
        if referenced_index is not None:
            # Add similarity scores of referenced movie
            for sim_idx, score in self.similarity[referenced_index]:
                movie_ranks[sim_idx] += score * 3.0 # Highly weighted link

        # Weight 2: Text matching query words against movie tags
        query_words = query.split()
        for idx, row in self.movies.iterrows():
            tags = str(row.get('tags', '')).lower()
            
            # Score matches on query words
            match_count = sum(1 for word in query_words if word in tags)
            movie_ranks[idx] += match_count * 0.1
            
            # Score matches on active genre keywords
            for key in active_keywords:
                if key in tags:
                    movie_ranks[idx] += 0.5

        # Avoid recommending the referenced movie itself
        if referenced_index is not None:
            movie_ranks[referenced_index] = -1.0

        # Sort and select top matching movies
        top_indices = np.argsort(movie_ranks)[::-1][:limit]
        
        recommendations = []
        for idx in top_indices:
            if movie_ranks[idx] <= 0:
                continue
            row = self.movies.iloc[idx]
            
            # Create a smart explanation based on matched attributes
            matched_aspects = []
            if referenced_index is not None:
                matched_aspects.append(f"similar to **{self.movies.iloc[referenced_index].title}**")
            for key in active_keywords:
                matched_aspects.append(f"**{key}** themes")
            
            if matched_aspects:
                explanation = "Matched " + " and ".join(matched_aspects)
            else:
                explanation = "Matched search terms"

            recommendations.append({
                "movie_id": row.movie_id,
                "title": row.title,
                "year": row.year,
                "vote_average": row.vote_average,
                "overview": row.get('original_overview', ''),
                "genres": row.get('original_genres', []),
                "explanation": explanation
            })

        # Backfill if nothing matched
        if not recommendations:
            trending = self.movies.sort_values(by='vote_average', ascending=False)
            for _, row in trending.head(limit).iterrows():
                recommendations.append({
                    "movie_id": row.movie_id,
                    "title": row.title,
                    "year": row.year,
                    "vote_average": row.vote_average,
                    "overview": row.get('original_overview', ''),
                    "genres": row.get('original_genres', []),
                    "explanation": "Trending hit"
                })

        return recommendations
