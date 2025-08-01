import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 🔧 Build path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "experiments.db")

# SQLite-Datenbank
engine = create_engine(f"sqlite:///{DB_PATH}")

SessionLocal = sessionmaker(bind=engine)

