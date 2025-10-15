# load_to_sql_server.py
import pandas as pd
from sqlalchemy import create_engine, text
import urllib

# --- Configuration: IMPORTANT! ---
# Replace with your actual server name from SSMS.
SERVER_NAME = "localhost\\SQLEXPRESS" 
DATABASE_NAME = "RestaurantReco" 
TABLE_NAMES = [
    'Users', 'Restaurants', 'VisitHistory', 'user_clusters', 
    'MedicalCondition', 'RestaurantMenu', 'UserMedicalCondition', 'VisitMenuItem'
]

# --- 1. Create the Database Connection Engine ---
# This connection string uses Windows Authentication (Trusted_Connection=yes).
# It's the standard way to connect on a local development machine.

# DB connection
engine = create_engine(
    'mssql+pyodbc://@localhost\\SQLEXPRESS/RestaurantReco?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server',
    pool_recycle=3600
)
print(f"Successfully created engine to connect to '{DATABASE_NAME}'.")

# --- 2. Loop Through CSVs and Load into SQL Server ---
with engine.connect() as connection:
    for table in TABLE_NAMES:
        csv_file = f"{table}.csv"
        try:
            df = pd.read_csv(csv_file)
            
            # Use pandas' to_sql method. This is powerful and efficient.
            # 'if_exists='replace'' will drop the table if it already exists and create a new one.
            df.to_sql(table, con=connection, if_exists='replace', index=False)
            
            print(f"✅ Successfully loaded '{csv_file}' into SQL Server table '{table}'.")
        except FileNotFoundError:
            print(f"⚠️ Warning: Could not find '{csv_file}'. Skipping.")
        except Exception as e:
            print(f"❌ Error loading '{csv_file}': {e}")

print("\nDatabase loading process complete.")