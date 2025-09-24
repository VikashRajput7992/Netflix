"""
Netflix Recommendation System - Data Processing Utilities

This module contains utility functions for data preprocessing,
feature engineering, and data management.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Data processing utilities for Netflix recommendation system"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.cleaned_dir = self.data_dir / "cleaned"
        self.final_dir = self.data_dir / "final"
        
    def load_raw_data(self):
        """Load raw Netflix movies data"""
        raw_file = self.raw_dir / "netflix_movies_detailed_up_to_2025.csv"
        
        if not raw_file.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_file}")
            
        return pd.read_csv(raw_file)
    
    def preprocess_movies_data(self, df):
        """Clean and preprocess the movies dataset"""
        df = df.copy()
        
        # Handle missing values
        df['vote_average'].fillna(df['vote_average'].median(), inplace=True)
        df['vote_count'].fillna(0, inplace=True)
        df['popularity'].fillna(df['popularity'].median(), inplace=True)
        df['budget'].fillna(0, inplace=True)
        df['revenue'].fillna(0, inplace=True)
        df['release_year'].fillna(df['release_year'].median(), inplace=True)
        
        # Fill categorical data
        df['director'].fillna('Unknown', inplace=True)
        df['cast'].fillna('Unknown', inplace=True)
        df['country'].fillna('Unknown', inplace=True)
        df['language'].fillna('en', inplace=True)
        df['genres'].fillna('Unknown', inplace=True)
        df['description'].fillna('No description', inplace=True)
        
        # Extract duration in minutes
        df['duration_minutes'] = df['duration'].apply(self._extract_duration)
        
        # Create binary features
        df['is_movie'] = (df['type'] == 'Movie').astype(int)
        df['has_budget'] = (df['budget'] > 0).astype(int)
        df['has_revenue'] = (df['revenue'] > 0).astype(int)
        
        # Create calculated features
        df['profit'] = df['revenue'] - df['budget']
        df['roi'] = np.where(df['budget'] > 0, df['profit'] / df['budget'], 0)
        df['movie_age'] = 2025 - df['release_year']
        df['popularity_rank'] = df['popularity'].rank(pct=True)
        
        return df
    
    def _extract_duration(self, duration_str):
        """Extract numeric duration from string"""
        if pd.isna(duration_str):
            return 120  # Default duration
                                            
        duration_str = str(duration_str)
        if 'min' in duration_str:
            try:
                return int(duration_str.replace(' min', ''))
            except:
                return 120
        elif 'Season' in duration_str:
            return 90  # Default for TV shows
        else:
            return 120
    
    def create_synthetic_users(self, n_users=5000):
        """Create synthetic user profiles"""
        np.random.seed(42)
        
        # User demographics
        ages = np.random.normal(35, 12, n_users).astype(int)
        ages = np.clip(ages, 18, 80)
        
        genders = np.random.choice(['Male', 'Female', 'Other'], n_users, p=[0.48, 0.48, 0.04])
        
        countries = np.random.choice(['USA', 'UK', 'Canada', 'Germany', 'France', 'Japan', 
                                     'Australia', 'Brazil', 'Mexico', 'India'], 
                                    n_users, p=[0.3, 0.15, 0.1, 0.08, 0.08, 0.05, 
                                               0.05, 0.05, 0.05, 0.09])
        
        subscription_types = np.random.choice(['Basic', 'Standard', 'Premium'], 
                                            n_users, p=[0.2, 0.5, 0.3])
        
        # Create preferred genres
        genre_options = ['Comedy', 'Drama', 'Action', 'Thriller', 'Horror', 'Romance', 
                        'Sci-Fi', 'Documentary', 'Animation', 'Fantasy']
        
        preferred_genres = []
        for _ in range(n_users):
            n_genres = np.random.randint(1, 4)
            user_genres = np.random.choice(genre_options, n_genres, replace=False)
            preferred_genres.append(','.join(user_genres))
        
        # Create DataFrame
        df_users = pd.DataFrame({
            'user_id': range(1, n_users + 1),
            'age': ages,
            'gender': genders,
            'country': countries,
            'subscription_type': subscription_types,
            'preferred_genres': preferred_genres,
            'avg_daily_watch_time': np.round(np.random.exponential(2.5, n_users), 2),
            'preferred_watch_time': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'],
                                                   n_users, p=[0.1, 0.2, 0.5, 0.2]),
            'account_age_months': np.random.randint(1, 60, n_users)
        })
        
        return df_users
    
    def create_user_movie_interactions(self, df_users, df_movies, n_interactions=50000):
        """Create synthetic user-movie interactions"""
        np.random.seed(42)
        
        movie_ids = df_movies['show_id'].tolist()
        user_ids = df_users['user_id'].tolist()
        
        interactions = []
        
        for _ in range(n_interactions):
            user_id = np.random.choice(user_ids)
            movie_id = np.random.choice(movie_ids)
            
            # Get user and movie info
            user_info = df_users[df_users['user_id'] == user_id].iloc[0]
            movie_info = df_movies[df_movies['show_id'] == movie_id].iloc[0]
            
            # Simulate rating based on preferences
            base_rating = movie_info['vote_average'] if pd.notna(movie_info['vote_average']) else 6.0
            
            # Genre matching bonus
            user_genres = user_info['preferred_genres'].split(',')
            movie_genres = movie_info['genres'] if pd.notna(movie_info['genres']) else ''
            
            genre_match_bonus = 0
            for user_genre in user_genres:
                if user_genre.lower() in movie_genres.lower():
                    genre_match_bonus += 0.5
            
            # Calculate final rating
            random_factor = np.random.normal(0, 1)
            final_rating = base_rating + genre_match_bonus + random_factor
            final_rating = np.clip(final_rating, 1, 10)
            
            # Convert to binary like/dislike
            liked = 1 if final_rating >= 6.5 else 0
            
            interactions.append({
                'user_id': user_id,
                'show_id': movie_id,
                'rating': round(final_rating, 1),
                'liked': liked,
                'interaction_date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
            })
        
        df_interactions = pd.DataFrame(interactions)
        
        # Remove duplicates
        df_interactions = df_interactions.drop_duplicates(subset=['user_id', 'show_id'], keep='first')
        
        return df_interactions
    
    def save_processed_data(self, df_movies=None, df_users=None, df_interactions=None):
        """Save processed data to cleaned folder"""
        self.cleaned_dir.mkdir(exist_ok=True)
        
        if df_movies is not None:
            df_movies.to_csv(self.cleaned_dir / "movies_cleaned.csv", index=False)
            print(f"✅ Saved cleaned movies data: {len(df_movies)} records")
            
        if df_users is not None:
            df_users.to_csv(self.cleaned_dir / "users_synthetic.csv", index=False)
            print(f"✅ Saved synthetic users data: {len(df_users)} records")
            
        if df_interactions is not None:
            df_interactions.to_csv(self.cleaned_dir / "interactions_synthetic.csv", index=False)
            print(f"✅ Saved synthetic interactions: {len(df_interactions)} records")

class FeatureEngineering:
    """Feature engineering utilities"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def engineer_features(self, df_interactions, df_users, df_movies):
        """Create comprehensive features for ML modeling"""
        
        # Merge datasets
        df_full = df_interactions.merge(df_users, on='user_id', how='left')
        df_full = df_full.merge(df_movies, on='show_id', how='left')
        
        # User statistics
        user_stats = df_interactions.groupby('user_id').agg({
            'rating': ['mean', 'std', 'count'],
            'liked': ['sum', 'mean']
        }).round(3)
        
        user_stats.columns = ['user_avg_rating', 'user_rating_std', 'user_total_interactions',
                             'user_total_likes', 'user_like_rate']
        user_stats['user_rating_std'].fillna(0, inplace=True)
        
        # Movie statistics
        movie_stats = df_interactions.groupby('show_id').agg({
            'rating': ['mean', 'std', 'count'],
            'liked': ['sum', 'mean']
        }).round(3)
        
        movie_stats.columns = ['movie_avg_rating', 'movie_rating_std', 'movie_total_interactions',
                              'movie_total_likes', 'movie_like_rate']
        movie_stats['movie_rating_std'].fillna(0, inplace=True)
        
        # Add statistics to main dataset
        df_full = df_full.merge(user_stats, on='user_id', how='left')
        df_full = df_full.merge(movie_stats, on='show_id', how='left')
        
        # Genre matching features
        df_full['genre_match_score'] = df_full.apply(self._calculate_genre_match, axis=1)
        
        # Derived features
        df_full['user_movie_rating_diff'] = df_full['user_avg_rating'] - df_full['vote_average']
        df_full['user_selectivity'] = 1 / (df_full['user_like_rate'] + 0.1)
        df_full['movie_appeal'] = df_full['movie_like_rate'] * df_full['vote_average']
        
        return df_full
    
    def _calculate_genre_match(self, row):
        """Calculate genre match score between user preferences and movie genres"""
        try:
            user_genres = str(row['preferred_genres']).lower().split(',')
            movie_genres = str(row['genres']).lower().split(',')
            
            user_genres = [g.strip() for g in user_genres if g.strip()]
            movie_genres = [g.strip() for g in movie_genres if g.strip()]
            
            if len(user_genres) > 0 and len(movie_genres) > 0:
                match_count = len(set(user_genres) & set(movie_genres))
                return match_count / len(user_genres)
            else:
                return 0
        except:
            return 0
    
    def prepare_features(self, df_features, target_col='liked'):
        """Prepare features for machine learning"""
        
        # Exclude non-feature columns
        exclude_columns = ['user_id', 'show_id', 'rating', 'liked', 'interaction_date',
                          'title', 'director', 'cast', 'country', 'date_added', 'description',
                          'type', 'genres', 'preferred_genres', 'language', 'gender', 
                          'subscription_type', 'preferred_watch_time', 'duration']
        
        # Get feature columns
        feature_cols = [col for col in df_features.columns if col not in exclude_columns]
        
        # Prepare X and y
        X = df_features[feature_cols].copy()
        y = df_features[target_col].copy()
        
        # Handle missing values
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            X[col].fillna(X[col].median(), inplace=True)
        
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            mode_val = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
            X[col].fillna(mode_val, inplace=True)
            
            # Label encoding
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[col].transform(X[col])
        
        return X, y, feature_cols