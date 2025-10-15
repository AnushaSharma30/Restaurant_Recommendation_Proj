import pandas as pd
import numpy as np

class RecommendationEngine:
    def __init__(self):
        """
        Initializes the engine by loading all necessary data files.
        """
        print("Initializing the Recommendation Engine...")
        try:
            self.users_df = pd.read_csv('Users.csv')
            self.restaurants_df = pd.read_csv('Restaurants.csv')
            self.visits_df = pd.read_csv('VisitHistory.csv')
            self.user_clusters = pd.read_csv('user_clusters.csv').set_index('user_id')
            print("Engine initialized successfully.")
        except FileNotFoundError as e:
            print(f"ERROR: Could not load a required file. {e}")
            self.users_df = None

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Helper function to calculate distance between two points in km."""
        R = 6371  # Radius of Earth in kilometers
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def get_recommendations(self, user_id, n_recommendations=5, distance_limit_km=5):
        """
        Generates a ranked list of restaurant recommendations for a given user,
        respecting the distance constraint.
        """
        if self.users_df is None: return None
            
        print(f"\n--- Generating recommendations for user_id: {user_id} ---")
        
        # --- 1. Get User Location ---
        try:
            user_location = self.users_df.loc[self.users_df['user_id'] == user_id, ['latitude', 'longitude']].iloc[0]
            user_lat, user_lon = user_location['latitude'], user_location['longitude']
        except IndexError:
            print(f"ERROR: Could not find location for user {user_id}. Cannot provide recommendations.")
            return None

        # --- 2. Pre-filter Restaurants by Distance (THE CRITICAL NEW STEP) ---
        distances = self._haversine_distance(user_lat, user_lon, self.restaurants_df['latitude'], self.restaurants_df['longitude'])
        nearby_restaurants = self.restaurants_df[distances <= distance_limit_km].copy()
        
        if nearby_restaurants.empty:
            print(f"No restaurants found within {distance_limit_km}km. Cannot provide recommendations.")
            return None
        
        print(f"Found {len(nearby_restaurants)} restaurants within {distance_limit_km}km.")
        
        # --- 3. Determine the user's cluster ---
        try:
            user_cluster_id = self.user_clusters.loc[user_id, 'cluster_id']
            print(f"User belongs to Cluster (Taste Tribe): {user_cluster_id}")
        except KeyError:
            user_cluster_id = -1
            print(f"User {user_id} not found in cluster data. Defaulting to non-personalized strategy.")
            
        # --- 4. Execute Recommendation Strategy ON NEARBY RESTAURANTS ONLY ---
        if user_cluster_id in [-1, 46]:
            print("Strategy: Ranking nearby restaurants by rating (Non-Personalized).")
            recommendations = nearby_restaurants.sort_values(by='rating', ascending=False)
        else:
            print(f"Strategy: Finding restaurants popular within Cluster {user_cluster_id} from the nearby pool (Personalized).")
            tribe_members = self.user_clusters[self.user_clusters['cluster_id'] == user_cluster_id].index.tolist()
            tribe_visits = self.visits_df[self.visits_df['user_id'].isin(tribe_members)]
            
            # Filter tribe visits to only include the nearby restaurants
            tribe_visits_nearby = tribe_visits[tribe_visits['restaurant_id'].isin(nearby_restaurants['restaurant_id'])]

            if tribe_visits_nearby.empty:
                print("Warning: No visit history found for this cluster in the local area. Defaulting to non-personalized strategy.")
                recommendations = nearby_restaurants.sort_values(by='rating', ascending=False)
            else:
                tribe_preferences = tribe_visits_nearby.groupby('restaurant_id').agg(
                    visit_count=('visit_id', 'count'),
                    avg_rating=('rating_given', 'mean')
                ).reset_index()
                
                # Merge with the nearby restaurants to get full details
                recommendations = pd.merge(nearby_restaurants, tribe_preferences, on='restaurant_id')
                recommendations = recommendations.sort_values(by=['visit_count', 'avg_rating'], ascending=False)

        # --- 5. Finalize and return ---
        user_visited_restaurants = self.visits_df[self.visits_df['user_id'] == user_id]['restaurant_id'].unique()
        final_recommendations = recommendations[~recommendations['restaurant_id'].isin(user_visited_restaurants)]
        
        return final_recommendations.head(n_recommendations)

# ==============================================================================
# --- EXAMPLE USAGE ---
# ==============================================================================
# ==============================================================================
# --- EXAMPLE USAGE ---
# ==============================================================================
if __name__ == '__main__':
    engine = RecommendationEngine()
    
    if engine.users_df is not None:
        
        # --- YOUR CUSTOM TEST ---
        # Simply change this number to any user ID you want to test
        
        my_user_id_to_test =700 # <--- CHANGE THIS VALUE
        
        recommendations = engine.get_recommendations(my_user_id_to_test)
        
        print(f"\n--- Recommendations for User {my_user_id_to_test} ---")
        print(recommendations)
        print("-" * 40)

        # You can add more tests as well
        another_user_id = 666
        more_recs = engine.get_recommendations(another_user_id)
        print(f"\n--- Recommendations for User {another_user_id} ---")
        print(more_recs)
        print("-" * 40)

    if engine.users_df is not None:
        # Test User 6 again, the recommendations should now be local
        user_6_recs = engine.get_recommendations(6)
        print("\n--- Recommendations for User 6 (Personalized & Local) ---")
        print(user_6_recs)
        print("-" * 55)

        # Test User 1 again, recommendations should still be local
        user_1_recs = engine.get_recommendations(1)
        print("\n--- Recommendations for User 1 (Non-Personalized & Local) ---")
        print(user_1_recs)
        print("-" * 55)