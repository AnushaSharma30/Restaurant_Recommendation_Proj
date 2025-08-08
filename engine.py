# engine.py (The Final, Case-Corrected Version)

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os

class RecommendationEngine:
    def __init__(self):
        """
        Initializes the engine by connecting to the database using a URL
        from environment variables.
        """
        print("Initializing the Recommendation Engine...")
        try:
            db_url = os.environ.get('DATABASE_URL')
            if not db_url:
                raise ValueError("DATABASE_URL environment variable not set.")
            
            self.db_engine = create_engine(db_url)
            
            # Load the full restaurants table into memory once, using the correct case
            with self.db_engine.connect() as conn:
                self.restaurants_df = pd.read_sql_table("Restaurants", conn)
            
            print("✅ Engine connected to PostgreSQL and initialized successfully.")
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Could not initialize engine. {e}")
            self.db_engine = None

    def _get_df_from_sql(self, query):
        """Helper to run a query and return a pandas DataFrame."""
        with self.db_engine.connect() as conn:
            return pd.read_sql_query(query, conn)

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def get_recommendations(self, user_id, n_recommendations=5, distance_limit_km=5):
        if self.db_engine is None: return None
        
        # --- 1. Get User Location & Cluster from DB (using correct case) ---
        query = f"""
            SELECT u.latitude, u.longitude, uc.cluster_id
            FROM "Users" u
            LEFT JOIN user_clusters uc ON u.user_id = uc.user_id
            WHERE u.user_id = {user_id}
        """
        user_data = self._get_df_from_sql(query)
        if user_data.empty: return None
        
        user_lat, user_lon, user_cluster_id = user_data.iloc[0]
        user_cluster_id = int(user_cluster_id) if pd.notna(user_cluster_id) else -1
        
        # --- 2. Pre-filter Restaurants by Distance ---
        distances = self._haversine_distance(user_lat, user_lon, self.restaurants_df['latitude'], self.restaurants_df['longitude'])
        nearby_restaurants = self.restaurants_df[distances <= distance_limit_km].copy()
        if nearby_restaurants.empty: return pd.DataFrame()
        
        # --- 3. Execute Recommendation Strategy ---
        if user_cluster_id in [-1, 46]:
            recommendations = nearby_restaurants.sort_values(by='rating', ascending=False)
        else:
            tribe_members_query = f"SELECT user_id FROM user_clusters WHERE cluster_id = {user_cluster_id}"
            nearby_rest_ids = ','.join(map(str, nearby_restaurants['restaurant_id']))
            
            tribe_visits_query = f"""
                SELECT 
                    restaurant_id, 
                    COUNT(visit_id) as visit_count, 
                    AVG(rating_given) as avg_rating
                FROM "VisitHistory"
                WHERE user_id IN ({tribe_members_query})
                  AND restaurant_id IN ({nearby_rest_ids})
                GROUP BY restaurant_id
            """
            tribe_preferences = self._get_df_from_sql(tribe_visits_query)
            
            if tribe_preferences.empty:
                recommendations = nearby_restaurants.sort_values(by='rating', ascending=False)
            else:
                recommendations = pd.merge(nearby_restaurants, tribe_preferences, on='restaurant_id').sort_values(by=['visit_count', 'avg_rating'], ascending=False)

        # --- 4. Finalize and return ---
        user_visited_query = f'SELECT DISTINCT restaurant_id FROM "VisitHistory" WHERE user_id = {user_id}'
        user_visited_df = self._get_df_from_sql(user_visited_query)
        user_visited_restaurants = user_visited_df['restaurant_id'].unique()
        final_recommendations = recommendations[~recommendations['restaurant_id'].isin(user_visited_restaurants)]
        
        return final_recommendations.head(n_recommendations)