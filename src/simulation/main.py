import pandas as pd

from classes.Workflow import JobOperationWorkflowCollection
from src.simulation.ProductionSimulation import ProductionSimulation

if __name__ == "__main__":

    from configs.path_manager import get_path

    basic_data_path = get_path("data", "examples")
    df_schedule = pd.read_csv(basic_data_path / "lateness_schedule_day_01.csv")

    print("Maschinenbelegungsplan:")
    print(df_schedule.head(5))
    print("\n", "---" * 60)


    schedule_collection = JobOperationWorkflowCollection.from_dataframe(df_schedule)


    #for job_id, operations in schedule_collection.items():
    #    print(job_id, operations)

    simulation = ProductionSimulation(shift_length=1440, sigma= 0.02)

    simulation.run(schedule_collection, end_time= 1200)

    print(simulation.get_finished_operations().to_dataframe().head(5))
