import os
from peewee import SqliteDatabase

# Basisverzeichnis = Speicherort dieses Scripts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Datenbankpfade
PRODUCTION_DB_PATH = os.path.join(BASE_DIR, "production.db")
ANALYSIS_DB_PATH = os.path.join(BASE_DIR, "analysis_basics.db")

# Verbindungen
production_db = SqliteDatabase(PRODUCTION_DB_PATH)
analysis_db = SqliteDatabase(ANALYSIS_DB_PATH)

