import time

import pandas as pd
from matplotlib import pyplot as plt

from src.Logger import Logger

# DataFrame analyses
from src.DataFrameAnalyses import DataFramePlotGenerator, DataFrameChecker

# Domain data
from src.domain.Query import RoutingQuery
from src.domain.Initializer import JobsInitializer
from src.domain.Collection import LiveJobCollection

# Solver
from src.solvers.CP_Solver import Solver

output_path = "output"
logger = Logger(log_file = f"{output_path}/projektseminar_CP_Makespan.log")

if __name__ == "__main__":


    # Solver limits
    max_solver_time = 60 * 3  # 3 minutes

    # I. Data set

    # a) Load routing
    routings = RoutingQuery.get_by_source_name(source_name="Fisher and Thompson 10x10")

    # b) Create jobs from routings (without any change)
    jobs = JobsInitializer.create_simple_jobs(routings=routings, shuffle=False)

    jobs_collection = LiveJobCollection(jobs)
    print("-"*30, "Job-Shop Scheduling Problem", "-"*30)
    print(jobs_collection.to_operations_dataframe()[["Job", "Operation", "Machine", "Processing Time"]].head(10))
    print("-"*90)


    # II. Scheduling
    print("Scheduling ...")
    solver = Solver(
        jobs_collection=jobs_collection,
        schedule_start=0,
        logger=logger
    )
    solver.build_makespan_model()
    solver.log_model_info()

    solver.solve_model(
        log_file=f"{output_path}/optimize_makespan_cp_solver.log",
    )
    solver.log_solver_info()

    time.sleep(2)
    print("-" * 30, "Schedule", "-" * 30)
    schedule_job_collection = solver.get_schedule()
    df_schedule = schedule_job_collection.to_operations_dataframe()
    print(df_schedule[["Job", "Operation", "Machine", "Start", "Processing Time", "End"]].head(), end="\n\n")

    fig = DataFramePlotGenerator.get_gantt_chart_figure(df_schedule, perspective="Machine")
    fig.savefig(f"{output_path}/optimize_makespan_cp_gantt_chart.png", dpi=300)
    plt.show()

    DataFrameChecker.check_core_schedule_constraints(df_schedule)













