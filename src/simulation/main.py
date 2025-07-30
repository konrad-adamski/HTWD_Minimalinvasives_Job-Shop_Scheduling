import pandas as pd

from classes.Workflow import JobOperationWorkflowCollection
from src.simulation.ProductionRollingSimulation import ProductionSimulation

if __name__ == "__main__":

    from configs.path_manager import get_path

    basic_data_path = get_path("data", "examples")
    df_schedule = pd.read_csv(basic_data_path / "lateness_schedule_day_01.csv")

    print("Maschinenbelegungsplan:")
    print(df_schedule.head(5))
    print("\n", "---" * 60)


    schedule_collection = JobOperationWorkflowCollection.from_dataframe(df_schedule)
    for job_id, ops in schedule_collection.items():
        print(f"Job: {job_id}")
        for op in ops:
            print(f" {op.job_id} Seq {op.sequence_number}, Machine {op.machine} Start {op.start_time}, Duration {op.duration}, End {op.end_time}")
