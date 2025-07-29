import pandas as pd

# Data access
from configs.path_manager import get_path

# Database
from database.adder import add_routings_and_operations_from_dframe, add_jobs_from_dataframe
from database.db_models import Instance
from database.db_setup import SessionLocal, reset_db

if __name__ == '__main__':
    basic_data_path = get_path("data", "basic")

    reset_db()

    # ------ Instance --------------------

    instance_name = "Fisher and Thompson 10x10"

    with SessionLocal() as session:
        instance = Instance(name=instance_name)
        session.add(instance)
        session.commit()
        print(f"Instance created with name: '{instance.name}' (ID {instance.id})")


    # ----- Routings with operations -----
    df_routings = pd.read_csv(basic_data_path / "ft10_routings.csv")
    add_routings_and_operations_from_dframe(df_routings, instance.name)

    # ----- Jobs (50 days) ---------------
    df_jobs_times_final = pd.read_csv(basic_data_path / "ft10_jobs_times.csv")
    df_jobs_times = df_jobs_times_final[df_jobs_times_final["Ready Time"] <= 60 * 24 * 50]
    add_jobs_from_dataframe(df_jobs_times)





