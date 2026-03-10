import sqlite3
import pandas as pd
import os

# 1. Créer la connexion à la base de données
# Si le fichier "olist.db" n'existe pas, SQLite va le créer automatiquement.
conn = sqlite3.connect('olist.db')

# 2. Dictionnaire faisant le lien entre le nom du fichier CSV et le nom de la table SQL
fichiers_csv = {
    'olist_orders_dataset.csv': 'orders',
    'olist_customers_dataset.csv': 'customers',
    'olist_order_items_dataset.csv': 'order_items',
    'olist_products_dataset.csv': 'products'
}

# 3. Boucle pour importer chaque fichier
for fichier, nom_table in fichiers_csv.items():
    chemin_fichier = f"data/{fichier}"
    
    if os.path.exists(chemin_fichier):
        print(f"Importation de {fichier} dans la table '{nom_table}'...")
        # Lire le fichier CSV
        df = pd.read_csv(chemin_fichier)
        
        # Transférer le DataFrame vers SQLite
        # if_exists='replace' permet d'écraser la table si tu relances le script
        df.to_sql(nom_table, conn, if_exists='replace', index=False)
        print(f"✓ Table '{nom_table}' créée avec succès !")
    else:
        print(f"Attention : Le fichier {chemin_fichier} est introuvable.")

# 4. Fermer la connexion proprement
conn.close()
print("Terminé ! La base de données 'olist.db' est prête à être interrogée.")