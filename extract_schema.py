import sqlite3

def extract_database_schema(db_path: str, output_path: str):
    """
    Extracts the full schema (DDL statements) from the SQLite database.
    This schema will be injected into the LLM's system prompt so it understands
    the table structures and column names for accurate Text-to-SQL generation.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query the sqlite_master table to get all CREATE TABLE statements
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema_statements = []
    for table in tables:
        if table[0]:  # Ensure the SQL statement exists
            schema_statements.append(table[0])

    # Close the database connection
    conn.close()
    
    # Format the output with clear separation between tables
    full_schema_text = "\n\n".join(schema_statements)
    
    # Save the schema to a text file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_schema_text)
        
    print(f"✓ Schema successfully extracted and saved to '{output_path}'.")

if __name__ == "__main__":
    DB_FILE = 'olist.db'
    OUTPUT_FILE = 'schema.txt'
    
    extract_database_schema(DB_FILE, OUTPUT_FILE)