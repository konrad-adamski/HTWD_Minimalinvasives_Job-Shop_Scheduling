import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.db_models import Base

# 🔧 Build path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "experiments.db")

# ✅ Use absolute file path (prefix with 'sqlite:///')
engine = create_engine(f"sqlite:///{DB_PATH}")

SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)


def reset_db():
    """
    Drops all tables and recreates them using the Base metadata.
    Use with caution – this will erase all data.
    """
    Base.metadata.drop_all(bind=engine)
    print("✅ All tables dropped.")
    Base.metadata.create_all(bind=engine)
    print("✅ All tables recreated.")