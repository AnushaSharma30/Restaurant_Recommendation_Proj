import google.generativeai as genai
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import time
import re # Import regex for cleaning
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("GENAI_API_KEY")

# Configure genai with the environment key
genai.configure(api_key=api_key)

# Initialize Faker
fake = Faker('en_IN') # NEW: Using Indian locale for names

# Configure Gemini API (Tier 1 usage only where necessary)
# IMPORTANT: Replace with your actual key

gemini_model = genai.GenerativeModel('models/gemini-1.5-pro')

# DB connection
engine = create_engine(
    'mssql+pyodbc://@localhost\\SQLEXPRESS/PROJECT2?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server',
    pool_recycle=3600
)

# --- CONSTANTS ---
NUM_USERS = 100_000
NUM_RESTAURANTS = 1_000
NUM_VISITS = 100_000
# REMOVED: NUM_MENU_ITEMS is now calculated dynamically
NUM_USER_PREFERENCES = 200_000
NUM_CITIES = 15
NUM_MEDICAL_CONDITIONS = 10

# CHANGED: Reduced and specified cuisines as requested
CUISINES = ['Indian', 'Chinese', 'Japanese']

# NEW: Predefined list of popular dishes as requested
PREDEFINED_DISHES = {
    'Indian': ['Aamras Poori', 'Boondi Raita', 'Pakora', 'Sarson da Saag & Makki di Roti', 'Pav Bhaji'],
    'Chinese': ['Chilli Paneer', 'Gobi Manchurian', 'Chilli Mushroom', 'American Chopsuey', 'Schezwan Fried Rice'],
    'Japanese': ['Sushi Rolls', 'Matcha Ice Cream', 'Udon Noodles', 'Tonkotsu Ramen', 'Gyoza']
}


# --- Gemini Helper and Data Generators ---

def generate_with_gemini(prompt):
    """Helper function to call Gemini API and clean the response."""
    try:
        response = gemini_model.generate_content(prompt)
        time.sleep(1.5) # Maintain a safe rate limit
        text = response.text.strip()
        # Clean up common markdown formatting
        if text.startswith("```python"):
            text = text[9:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        # Handle cases where the response is a variable assignment
        if '=' in text:
            text = text.split('=', 1)[-1].strip()
        return text
    except Exception as e:
        print(f"An error occurred with Gemini: {e}")
        return "[]" # Return an empty list string on error

def generate_cities(num_cities):
    ALLOWED_CITIES = [
        'Mangalore', 'Bangalore', 'Udupi', 'Pune', 'Mumbai',
        'Chennai', 'Mysore', 'Surat', 'Hyderabad', 'Varanasi',
        'Kozhikode', 'Kolkata', 'Trivandrum', 'Amritsar', 'Kochi'
    ]
    prompt = f"""Generate exactly {num_cities} Indian cities from this list:
    {ALLOWED_CITIES}
    With their exact coordinates as a Python dictionary like: {{'Mumbai': (19.0760, 72.8777)}}
    Rules: only these cities, correct names, 4+ decimal precision, return a valid Python dictionary string only."""
    return eval(generate_with_gemini(prompt))

def generate_medical_conditions(num_conditions):
    prompt = f"""Generate a Python list of {num_conditions} common medical conditions that affect dietary choices.
Include 'None' as the first element. Example: ['None', 'Diabetes', 'Hypertension']"""
    return eval(generate_with_gemini(prompt))

# NEW: Batch generation of restaurant names for efficiency
def generate_restaurant_names(num_names):
    prompt = f"""Generate a Python list of {num_names} creative and authentic-sounding Indian restaurant names.
    Examples: 'The Saffron Plate', 'Tadka Town', 'Masala Bay'.
    Return only the Python list."""
    return eval(generate_with_gemini(prompt))

# NEW: Batch generation of dish names for efficiency
def generate_dish_names_batch(cuisine, num_dishes):
    prompt = f"Generate a Python list of {num_dishes} diverse and popular {cuisine} dish names. Return only the Python list."
    return eval(generate_with_gemini(prompt))


def generate_users():
    users, user_medical = [], []
    for user_id in range(1, NUM_USERS + 1):
        name = fake.name()
        birth_year = random.randint(1950, 2007)
        gender = random.choice(['Male', 'Female'])
        city, lat, lon = generate_city_location()
        users.append([user_id, name, birth_year, gender, lat, lon, city])
        
        # Assign medical conditions
        num_conditions = random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]
        # Ensure we don't try to sample more conditions than exist (excluding 'None')
        available_conditions = len(medical_conditions) - 1
        condition_ids = random.sample(range(2, len(medical_conditions) + 1), min(num_conditions, available_conditions))
        
        if not condition_ids:
            user_medical.append([user_id, 1]) # 1 is the ID for 'None'
        else:
            for cid in condition_ids:
                user_medical.append([user_id, cid])
                
    return (
        pd.DataFrame(users, columns=['user_id', 'name', 'birth_year', 'gender', 'latitude', 'longitude', 'city']),
        pd.DataFrame(user_medical, columns=['user_id', 'condition_id'])
    )

def generate_city_location():
    city = random.choice(list(city_coords.keys()))
    base_lat, base_lon = city_coords[city]
    # Generate a random point within an approximate 5km radius (0.05 degrees)
    lat = round(base_lat + random.uniform(-0.05, 0.05), 6)
    lon = round(base_lon + random.uniform(-0.05, 0.05), 6)
    return city, lat, lon

def generate_restaurants(restaurant_names):
    restaurants = []
    for rest_id in range(1, NUM_RESTAURANTS + 1):
        # CHANGED: Use pre-generated Indian-style names
        name = random.choice(restaurant_names) + ' ' + random.choice(['Kitchen', 'Bistro', 'Cafe', 'Grill', 'House'])
        city, lat, lon = generate_city_location()
        rating = round(random.uniform(2.5, 5.0), 1)
        pet_friendly = random.random() > 0.7
        restaurants.append([rest_id, name.strip(), pet_friendly, lat, lon, city, rating])
    return pd.DataFrame(restaurants, columns=[
        'restaurant_id', 'name', 'pet_friendly', 'latitude', 'longitude', 'city', 'rating'
    ])

# CHANGED: Major overhaul of this function for efficiency and correctness
def generate_restaurant_menu(dish_name_pool):
    menu_items = []
    menu_id = 1
    
    # Assign a primary cuisine to each restaurant
    rest_cuisines = {rest_id: random.choice(CUISINES) for rest_id in range(1, NUM_RESTAURANTS + 1)}

    for rest_id in range(1, NUM_RESTAURANTS + 1):
        primary_cuisine = rest_cuisines[rest_id]
        dishes_for_menu = set() # Use a set to avoid duplicate dishes in one menu

        # 1. Add some predefined dishes for the restaurant's primary cuisine
        num_predefined = random.randint(2, len(PREDEFINED_DISHES[primary_cuisine]))
        predefined_sample = random.sample(PREDEFINED_DISHES[primary_cuisine], num_predefined)
        dishes_for_menu.update(predefined_sample)

        # 2. Add other random dishes from the generated pool for that cuisine
        num_random_dishes = random.randint(5, 15) # Each restaurant has 5-15 additional unique dishes
        random_sample = random.sample(dish_name_pool[primary_cuisine], min(num_random_dishes, len(dish_name_pool[primary_cuisine])))
        dishes_for_menu.update(random_sample)

        # 3. Create the menu item entries
        for dish in dishes_for_menu:
            # Simple heuristic for veg/vegan status
            is_veg = not any(meat in dish.lower() for meat in ['beef', 'chicken', 'pork', 'lamb', 'fish', 'tonkotsu'])
            is_vegan = is_veg and not any(dairy in dish.lower() for dairy in ['paneer', 'cheese', 'butter', 'ghee', 'ice cream'])
            
            menu_items.append([menu_id, rest_id, dish, primary_cuisine, is_veg, is_vegan])
            menu_id += 1

    return pd.DataFrame(menu_items, columns=[
        'menu_id', 'restaurant_id', 'dish_name', 'cuisine', 'is_vegetarian', 'is_vegan'
    ])

def generate_visit_history(menu_df):
    visits, visit_menu = [], []
    menu_by_rest = menu_df.groupby('restaurant_id')
    for visit_id in range(1, NUM_VISITS + 1):
        user_id = random.randint(1, NUM_USERS)
        rest_id = random.randint(1, NUM_RESTAURANTS)
        rating = round(random.uniform(1.0, 5.0), 1)
        visit_date = datetime.now() - timedelta(days=random.randint(0, 5*365))
        
        # Add feedback to ~30% of visits
        feedback = fake.sentence() if random.random() < 0.3 else None
        
        visits.append([visit_id, user_id, rest_id, rating, visit_date.date(), feedback])
        
        try:
            rest_menu = menu_by_rest.get_group(rest_id)
            # A user orders between 1 and 4 items per visit
            num_items_ordered = random.randint(1, min(4, len(rest_menu)))
            for _, item in rest_menu.sample(num_items_ordered).iterrows():
                visit_menu.append([visit_id, item['menu_id']])
        except (KeyError, ValueError):
            # Skip if restaurant has no menu (shouldn't happen now) or sample is empty
            continue
            
    return (
        pd.DataFrame(visits, columns=['visit_id', 'user_id', 'restaurant_id', 'rating_given', 'visit_date', 'feedback']),
        pd.DataFrame(visit_menu, columns=['visit_id', 'menu_id'])
    )

def generate_user_preferences(menu_df):
    user_prefs = []
    # 80% of users will have some preferences listed
    user_sample = random.sample(range(1, NUM_USERS + 1), int(NUM_USERS * 0.8))
    
    for user_id in user_sample:
        # Each user likes between 1 and 10 dishes
        num_prefs = random.randint(1, 10)
        for menu_id in menu_df.sample(num_prefs)['menu_id'].values:
            user_prefs.append([user_id, menu_id])
            if len(user_prefs) >= NUM_USER_PREFERENCES:
                break
        if len(user_prefs) >= NUM_USER_PREFERENCES:
            break
            
    return pd.DataFrame(user_prefs, columns=['user_id', 'menu_id'])

def get_master_table(items, id_col='id', name_col='name'):
    return pd.DataFrame([(i + 1, item) for i, item in enumerate(items)], columns=[id_col, name_col])

if __name__ == "__main__":
    print("🔁 Generating data using Gemini and local logic...")

    print("🌍 Generating cities...")
    city_coords = generate_cities(NUM_CITIES)

    print("💊 Generating medical conditions...")
    medical_conditions = generate_medical_conditions(NUM_MEDICAL_CONDITIONS)
    
    # --- NEW: Batch generation of names before main loops ---
    print("🏨 Generating Indian-style restaurant names...")
    restaurant_names_list = generate_restaurant_names(int(NUM_RESTAURANTS / 2)) # Generate half to allow for variety
    
    print("🍲 Generating dish names for all cuisines...")
    dish_name_pool = {}
    for cuisine in CUISINES:
        # Generate 200 unique dishes per cuisine for a rich variety
        dish_name_pool[cuisine] = generate_dish_names_batch(cuisine, 200)

    print("\n💾 Saving master tables and generating data...")
    
    # Master tables
    medical_df = get_master_table(medical_conditions, 'condition_id', 'name')
    medical_df.to_sql('MedicalCondition', engine, if_exists='replace', index=False)
    medical_df.to_csv("MedicalCondition.csv", index=False)
    
    # Note: A separate Cuisine table is good practice, but the schema uses a string in the menu.
    # Sticking to the implemented schema for now.

    print("👤 Generating users...")
    users_df, user_medical_df = generate_users()
    users_df.to_sql('Users', engine, if_exists='replace', index=False)
    users_df.to_csv("Users.csv", index=False)
    user_medical_df.to_sql('UserMedicalCondition', engine, if_exists='replace', index=False)
    user_medical_df.to_csv('UserMedicalCondition.csv', index=False)

    print("🍴 Generating restaurants...")
    restaurants_df = generate_restaurants(restaurant_names_list)
    restaurants_df.to_sql('Restaurants', engine, if_exists='replace', index=False)
    restaurants_df.to_csv('Restaurants.csv', index=False)

    print("📋 Generating menu...")
    menu_df = generate_restaurant_menu(dish_name_pool)
    menu_df.to_sql('RestaurantMenu', engine, if_exists='replace', index=False)
    menu_df.to_csv('RestaurantMenu.csv', index=False)
    
    print(f"Generated {len(menu_df)} total menu items across {NUM_RESTAURANTS} restaurants.")

    print("🧾 Generating visit history...")
    visits_df, visit_menu_df = generate_visit_history(menu_df)
    visits_df.to_sql('VisitHistory', engine, if_exists='replace', index=False)
    visits_df.to_csv('VisitHistory.csv', index=False)
    visit_menu_df.to_sql('VisitMenuItem', engine, if_exists='replace', index=False)
    visit_menu_df.to_csv('VisitMenuItem.csv', index=False)

    print("🎯 Generating user preferences...")
    user_prefs_df = generate_user_preferences(menu_df)
    user_prefs_df.to_sql('UserMenuPreference', engine, if_exists='replace', index=False)
    user_prefs_df.to_csv('UserMenuPreference.csv', index=False)

    print("\n✅ All tables generated and saved successfully.")