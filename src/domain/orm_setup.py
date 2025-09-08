import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, registry

from config.project_config import PROJECT_ROOT, get_data_path

# 🔧 Build path relative to this file

DB_PATH = os.path.join(PROJECT_ROOT, "experiments.db")
#DB_PATH = get_data_path(sub_directory="2025_09_08_Limit", file_name="experiments.db")


# SQLite-Datenbank
my_engine = create_engine(f"sqlite:///{DB_PATH}")

SessionLocal = sessionmaker(bind=my_engine)

# zentrale Registry
mapper_registry = registry()



def create_tables():
    mapper_registry.metadata.create_all(my_engine)


def reset_tables():
    confirmation = input(
        "⚠️ Are you sure you want to reset ALL tables? With 'yes' ALL DATA will be lost. [yes/No]\n"
    ).strip().lower().replace("'", "").replace('"', '')
    if confirmation != "yes" :
        print("❌ Operation cancelled. Nothing has been changed.")
        return

    mapper_registry.metadata.drop_all(my_engine)
    mapper_registry.metadata.create_all(my_engine)
    print("✅ All tables have been reset.")

