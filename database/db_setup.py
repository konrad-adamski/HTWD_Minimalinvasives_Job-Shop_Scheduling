import os
from sqlalchemy import create_engine, text
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
    create_job_operation_view()


def reset_db():
    """
    Drops all tables and the view, then recreates them.
    """
    with engine.connect() as conn:
        conn.execute(text("DROP VIEW IF EXISTS job_operation;"))
        print("🗑️ View 'job_operation' dropped (if existed).")

    Base.metadata.drop_all(bind=engine)
    print("✅ All tables dropped.")

    Base.metadata.create_all(bind=engine)
    print("✅ All tables recreated.")

    create_job_operation_view()



def create_job_operation_view():
    """
    Creates the SQL VIEW 'job_operation' if it does not already exist.
    Only works for SQLite or engines that support 'CREATE VIEW IF NOT EXISTS'.
    """
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE VIEW IF NOT EXISTS job_operation AS
            SELECT
                j.id AS job_id, j.routing_id, 
                rout_op.number AS operation_number, rout_op.machine, rout_op.duration,
                j.arrival, j.ready_time, j.deadline
            FROM job j
            JOIN routing_operation ro ON j.routing_id = rout_op.routing_id;
        """))
        print("✅ View 'job_operation' created (if not exists).")