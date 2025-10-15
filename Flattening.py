import pandas as pd
import numpy as np
import math
import gc
from tqdm import tqdm

def flatten_with_hybrid_craving():
    """
    Loads local CSVs, aggregates TRUE historical transaction data,
    and generates a new 'craving' feature based on both correlation
    and seasonal probabilities.
    """
    # ==============================================================================
    # 1. LOAD LOCAL CSV FILES
    # ==============================================================================
    print("Loading local CSV files...")
    try:
        users_df = pd.read_csv('Users.csv')
        restaurants_df = pd.read_csv('Restaurants.csv')
        user_medical_df = pd.read_csv('UserMedicalCondition.csv')
        menu_df = pd.read_csv('RestaurantMenu.csv')
        visits_df = pd.read_csv('VisitHistory.csv')
        visit_menu_df = pd.read_csv('VisitMenuItem.csv')
        medical_conditions_master_df = pd.read_csv('MedicalCondition.csv')
        print("All 8 CSV files loaded successfully.")
    except FileNotFoundError as e:
        print(f"ERROR: Could not find a required file: {e}. Make sure all 8 CSVs are in the same directory.")
        return

    # ==============================================================================
    # 2. PRE-COMPUTATION AND SETUP
    # ==============================================================================
    print("Preparing and pre-computing data...")

    # --- a. Identify Top 15 Dishes ---
    dish_visit_counts = pd.merge(visit_menu_df, menu_df, on='menu_id')
    top_15_dishes = dish_visit_counts['dish_name'].value_counts().nlargest(15).index.tolist()
    print("Top 15 dishes identified.")

    # --- b. User and Medical Data ---
    users_df['age'] = 2025 - users_df['birth_year']
    medical_conditions_list = medical_conditions_master_df[medical_conditions_master_df['name'] != 'None']['name'].tolist()
    user_medical_df_pivoted = user_medical_df.pivot_table(index='user_id', columns='condition_id', aggfunc='size', fill_value=0).reset_index()
    id_to_name_map = medical_conditions_master_df.set_index('condition_id')['name'].to_dict()
    user_medical_df_pivoted.rename(columns=id_to_name_map, inplace=True)
    for cond in medical_conditions_list:
        if cond not in user_medical_df_pivoted.columns:
            user_medical_df_pivoted[cond] = 0

    # --- c. Restaurant Selection ---
    visit_counts = visits_df.groupby(['user_id', 'restaurant_id']).size().reset_index(name='visit_count')
    top_restaurants = visit_counts.sort_values('visit_count', ascending=False).groupby('user_id').head(2)
    user_restaurant_pairs = []
    restaurants_by_city = restaurants_df.groupby('city')['restaurant_id'].apply(list).to_dict()
    all_rest_ids = restaurants_df['restaurant_id'].tolist()
    for user_id in tqdm(users_df['user_id'], desc="Pairing Users with Restaurants"):
        user_top_rests = top_restaurants[top_restaurants['user_id'] == user_id]['restaurant_id'].tolist()
        needed = 2 - len(user_top_rests)
        if needed > 0:
            try:
                user_city = users_df.loc[users_df['user_id'] == user_id, 'city'].iloc[0]
                city_restaurants = restaurants_by_city.get(user_city, [])
                available_rests = [r for r in city_restaurants if r not in user_top_rests]
                if len(available_rests) >= needed:
                    user_top_rests.extend(np.random.choice(available_rests, needed, replace=False))
                else:
                    user_top_rests.extend(available_rests)
                    needed_still = 2 - len(user_top_rests)
                    global_fallback = [r for r in all_rest_ids if r not in user_top_rests]
                    user_top_rests.extend(np.random.choice(global_fallback, needed_still, replace=False))
            except (IndexError, KeyError):
                global_fallback = [r for r in all_rest_ids if r not in user_top_rests]
                user_top_rests.extend(np.random.choice(global_fallback, needed, replace=False))
        for rest_id in user_top_rests:
            user_restaurant_pairs.append({'user_id': user_id, 'restaurant_id': rest_id})
    base_df = pd.DataFrame(user_restaurant_pairs)

    # --- d. Aggregate TRUE Historical Transaction Counts ---
    print("Aggregating historical transaction data...")
    visits_df['visit_date'] = pd.to_datetime(visits_df['visit_date'])
    visits_df['month'] = visits_df['visit_date'].dt.month
    
    full_history = visits_df.merge(visit_menu_df, on='visit_id').merge(menu_df[['menu_id', 'dish_name']], on='menu_id')
    
    full_history = full_history[full_history['dish_name'].isin(top_15_dishes)]
    transaction_counts = full_history.groupby(['user_id', 'restaurant_id', 'month', 'dish_name']).size().reset_index(name='order_count')
    historical_data_pivot = transaction_counts.pivot_table(
        index=['user_id', 'restaurant_id'], columns=['month', 'dish_name'], values='order_count', fill_value=0
    ).reset_index()
    
   
    # This new logic correctly handles the complex column names from the pivot table.
    new_cols = []
    for col in historical_data_pivot.columns.to_flat_index():
        if isinstance(col, tuple):
            # Check for columns from the index like ('user_id', '')
            if col[1] == '':
                new_cols.append(col[0])
            # Handle pivoted columns like (1, 'Pav Bhaji')
            else:
                new_cols.append(f"{col[0]}_{col[1]}")
        else:
            new_cols.append(col)
    historical_data_pivot.columns = new_cols
    
    # --- e. Assemble Final Base DataFrame ---
    print("Assembling final base DataFrame...")
    flat_df = pd.merge(base_df, historical_data_pivot, on=['user_id', 'restaurant_id'], how='left')
    flat_df = pd.merge(flat_df, users_df[['user_id', 'latitude', 'longitude', 'age']], on='user_id')
    flat_df = pd.merge(flat_df, restaurants_df[['restaurant_id', 'name', 'latitude', 'longitude']], on='restaurant_id')
    flat_df = pd.merge(flat_df, user_medical_df_pivoted, on='user_id', how='left')
    flat_df.rename(columns={'latitude_x': 'user_lat', 'longitude_x': 'user_long', 'name': 'rest_name', 'latitude_y': 'rest_lat', 'longitude_y': 'rest_long'}, inplace=True)
    flat_df[medical_conditions_list] = flat_df[medical_conditions_list].fillna(0).astype(int)

    # ==============================================================================
    # 3. GENERATE CRAVINGS AND WRITE IN CHUNKS
    # ==============================================================================
    OUTPUT_FILENAME = 'flattened_hybrid_craving_final.csv'
    CHUNK_SIZE = 20000
    num_chunks = math.ceil(len(flat_df) / CHUNK_SIZE)
    CRAVING_CORRELATION, BASE_ORDER_PROB, SEASONAL_BOOST = 0.75, 0.10, 0.40
    SEASONAL_DISHES = {
        'Summer': ['Aamras Poori', 'Boondi Raita', 'Sushi Rolls', 'Matcha Ice Cream', 'Chilli Paneer'],
        'Monsoon': ['Sarson da Saag & Makki di Roti', 'Pav Bhaji', 'Gyoza', 'Gobi Manchurian', 'Chilli Mushroom'],
        'Winter': ['Pakora', 'Udon Noodles', 'Tonkotsu Ramen', 'American Chopsuey', 'Schezwan Fried Rice']
    }

    def get_season(month):
        if month in [3, 4, 5, 6]: return 'Summer'
        if month in [7, 8, 9, 10]: return 'Monsoon'
        return 'Winter'

    print(f"\nGenerating hybrid cravings and writing to '{OUTPUT_FILENAME}'...")
    for i in tqdm(range(num_chunks), desc="Processing Chunks"):
        start_index = i * CHUNK_SIZE
        end_index = start_index + CHUNK_SIZE
        df_chunk = flat_df.iloc[start_index:end_index].copy()
        
        # --- Generate HYBRID Craving Data ---
        for month in range(1, 13):
            season = get_season(month)
            seasonal_dishes_list = SEASONAL_DISHES.get(season, [])
            
            for dish_idx, dish_name in enumerate(top_15_dishes, 1):
                transaction_col_name = f"{month}_{dish_name}"
                craving_col_name = f"month_{month}_craving_{dish_idx}"
                
                if transaction_col_name not in df_chunk.columns: df_chunk[transaction_col_name] = 0
                df_chunk[transaction_col_name].fillna(0, inplace=True)
                
                real_transactions = df_chunk[transaction_col_name]
                
                # Outcome 1: Correlated (same as transaction)
                correlated_craving = real_transactions
                
                # Outcome 2: Simulated based on season
                prob = BASE_ORDER_PROB + SEASONAL_BOOST if dish_name in seasonal_dishes_list else BASE_ORDER_PROB
                seasonal_craving = np.random.choice([0, 1, 2, 3], size=len(df_chunk), p=[1 - prob, prob * 0.6, prob * 0.3, prob * 0.1])
                
                # Decide which outcome to use based on the correlation probability
                use_correlated = np.random.rand(len(df_chunk)) < CRAVING_CORRELATION
                df_chunk[craving_col_name] = np.where(use_correlated, correlated_craving, seasonal_craving)

        # --- Finalize Column Names and Order ---
        transaction_cols = []
        for month in range(1, 13):
            for dish_idx, dish_name in enumerate(top_15_dishes, 1):
                new_name = f"month_{month}_menu_item_{dish_idx}"
                df_chunk.rename(columns={f"{month}_{dish_name}": new_name}, inplace=True)
                transaction_cols.append(new_name)

        user_cols, rest_cols = ['user_id', 'user_lat', 'user_long', 'age'], ['restaurant_id', 'rest_name', 'rest_lat', 'rest_long']
        craving_cols = [f'month_{m}_craving_{d}' for m in range(1, 13) for d in range(1, 16)]
        final_cols = user_cols + rest_cols + transaction_cols + craving_cols + medical_conditions_list
        
        for col in final_cols:
            if col not in df_chunk.columns: df_chunk[col] = 0
        result_chunk = df_chunk[final_cols]
        
        # --- Write to CSV ---
        if i == 0:
            result_chunk.to_csv(OUTPUT_FILENAME, index=False, mode='w', header=True)
        else:
            result_chunk.to_csv(OUTPUT_FILENAME, index=False, mode='a', header=False)
        gc.collect()

    print("\n=======================================================")
    print("✅ Flattening process with REAL history and HYBRID cravings complete!")
    print(f"File saved as: {OUTPUT_FILENAME}")
    print("=======================================================")

if __name__ == "__main__":
    flatten_with_hybrid_craving()