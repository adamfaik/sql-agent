import sqlite3
import pandas as pd
import os

def build_database(db_path: str, data_dir: str):
    """
    Reads multiple CSV files from the Olist E-commerce dataset and imports 
    them into a local SQLite database to enable complex SQL querying.
    """
    # Establish connection to the SQLite database
    conn = sqlite3.connect(db_path)
    
    # Dictionary mapping raw CSV filenames to clean SQL table names
    csv_files = {
        'olist_orders_dataset.csv': 'orders',
        'olist_customers_dataset.csv': 'customers',
        'olist_order_items_dataset.csv': 'order_items',
        'olist_products_dataset.csv': 'products',
        'olist_order_reviews_dataset.csv': 'order_reviews',
        'olist_sellers_dataset.csv': 'sellers',
        'olist_geolocation_dataset.csv': 'geolocation',
        'olist_order_payments_dataset.csv': 'order_payments',
        'product_category_name_translation.csv': 'category_translation'
    }

    # Iterate through the dictionary and ingest data
    for filename, table_name in csv_files.items():
        file_path = os.path.join(data_dir, filename)
        
        if os.path.exists(file_path):
            print(f"Loading {filename} into table '{table_name}'...")
            
            # Load CSV into a Pandas DataFrame
            df = pd.read_csv(file_path)
            
            # Write records stored in the DataFrame to the SQLite database
            # if_exists='replace' ensures idempotency (can be run multiple times)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            print(f"✓ Table '{table_name}' successfully created.")
        else:
            print(f"Warning: File {file_path} not found. Skipping...")

    # Close the database connection
    conn.close()
    print("\nDatabase build complete. The SQLite database is ready for querying.")

if __name__ == "__main__":
    DB_FILE = 'data/olist.db'
    DATA_DIRECTORY = 'data'
    
    build_database(DB_FILE, DATA_DIRECTORY)