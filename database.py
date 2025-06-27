import os
from peewee import SqliteDatabase

# Basisverzeichnis = Speicherort dieses Scripts (nicht Arbeitsverzeichnis!)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Absoluter Pfad zur DB im Projekt-Hauptverzeichnis
DB_PATH = os.path.join(BASE_DIR, "production.db")

# Verbindung
db = SqliteDatabase(DB_PATH)
