import sqlite3

def generate_enriched_schema(db_path="data/olist.db", output_path="data/schema.txt"):
    """
    Connects to the SQLite database, extracts the schema, 
    and automatically fetches sample values for text columns 
    to enrich the LLM prompt (Schema Linking).
    """
    print("⏳ Profiling database and generating enriched schema...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Retrieve all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema_lines = []

    for table_name_tuple in tables:
        table_name = table_name_tuple[0]
        schema_lines.append(f"CREATE TABLE {table_name} (")

        # Get column details using PRAGMA
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()

        col_lines = []
        for col in columns:
            col_name = col[1]
            col_type = col[2]

            comment = ""
            # If it's a TEXT column, we fetch up to 4 sample values for the LLM
            if col_type.upper() in ["TEXT", "VARCHAR"]:
                try:
                    cursor.execute(f"SELECT DISTINCT {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL LIMIT 4;")
                    samples = [str(row[0]) for row in cursor.fetchall()]
                    if samples:
                        # Format the samples like: 'val1', 'val2'
                        formatted_samples = ", ".join([f"'{s}'" for s in samples])
                        comment = f" -- Examples: {formatted_samples}"
                except Exception:
                    pass # Safely ignore if the sample query fails

            col_lines.append(f"    {col_name} {col_type}{comment}")

        # Join columns with a comma and close the table definition
        schema_lines.append(",\n".join(col_lines))
        schema_lines.append(");\n")

    # Write the dynamically generated schema to the text file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(schema_lines))

    print(f"✅ Enriched schema successfully generated in '{output_path}'.")
    conn.close()

if __name__ == "__main__":
    generate_enriched_schema()