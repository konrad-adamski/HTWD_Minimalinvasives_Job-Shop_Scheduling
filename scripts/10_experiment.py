from datetime import datetime
from decimal import Decimal

from project_config import get_data_path
from src.domain.Collection import LiveJobCollection
from src.domain.Initializer import ExperimentInitializer
from src.domain.Query import JobQuery, MachineQuery
from src.simulation.ProductionSimulation import ProductionSimulation
from src.solvers.CP_Solver import Solver

logs_path = get_data_path("solver_logs")

if __name__ == "__main__":
    source_name = "Fisher and Thompson 10x10"
    sim_sigma = 0.25
    absolute_lateness_ratio = 0.5
    inner_tardiness_ratio = 0.5
    max_bottleneck_utilization = 0.85
    total_shift_number = 5
    max_solver_time = 60 * 30 # 30min

    experiment_id = ExperimentInitializer.insert_experiment(
        source_name=source_name,
        absolute_lateness_ratio=absolute_lateness_ratio,
        inner_tardiness_ratio=inner_tardiness_ratio,
        max_bottleneck_utilization=Decimal(f"{max_bottleneck_utilization}"),
        sim_sigma=sim_sigma,
    )

    # TODO Get w_t w_e w_dev from experiment

    w_t = 3
    w_e = 1
    w_dev = 4


    # --- Preparation  ---
    simulation = ProductionSimulation(sigma=sim_sigma, verbose=False)

    # Jobs Collection
    jobs = JobQuery.get_by_source_name_max_util_and_lt_arrival(
        source_name=source_name,
        max_bottleneck_utilization=Decimal(f"{max_bottleneck_utilization}"),
        arrival_limit=60*24*total_shift_number
    )
    jobs_collection = LiveJobCollection(jobs)

    # Machines with transition times
    machines = MachineQuery.get_machines(
        source_name=source_name,
        max_bottleneck_utilization=Decimal(f"{max_bottleneck_utilization}"),
    )

    # Add transition times to operations
    for machine in machines:
        for job in jobs_collection.values():
            for operation in job.operations:
                if operation.machine_name == machine.name:
                    operation.transition_time = machine.transition_time

    # Collections (empty)
    schedule_jobs_collection = LiveJobCollection()  # pseudo previous schedule
    active_job_ops_collection = LiveJobCollection()

    waiting_job_ops_collection = LiveJobCollection()


    for shift_number in range(1, total_shift_number + 1):
        time_stamp = datetime.now().strftime("%Y-%m-%d_%H_%M")
        shift_start = shift_number* 1440
        shift_end = (shift_number+1) * 1440
        print(f"Shift number = {shift_number}: [{shift_start}, {shift_end}]")

        new_jobs_collection = jobs_collection.get_subset_by_earliest_start(earliest_start=shift_start)
        current_jobs_collection = new_jobs_collection + waiting_job_ops_collection



        # --- Scheduling ---
        solver = Solver(
            jobs_collection=current_jobs_collection,
            schedule_start=shift_start
        )

        solver.build_model__absolute_lateness__start_deviation__minimization(
            previous_schedule_jobs_collection=schedule_jobs_collection,
            active_jobs_collection=active_job_ops_collection,
            w_t=w_t, w_e=w_e, w_dev=w_dev                                                                                     # TODO params based on ratio
        )
        solver.print_model_info()
        solver.solve_model(
            gap_limit=0.02,
            time_limit=max_solver_time,
            log_file= f"{logs_path}/{time_stamp}_e{experiment_id}t{w_t}e{w_e}dev{w_dev}_sh{shift_number:02d}.log"
        )
        solver.print_solver_info()

        schedule_jobs_collection = solver.get_schedule()                                                                # TODO save in DB


        # --- Simulation ---
        simulation.run(
            schedule_collection=schedule_jobs_collection,
            start_time=shift_start,
            end_time=shift_end
        )

        simulation_jobs = simulation.get_finished_operation_collection()                                                # TODO save in DB

        active_job_ops_collection = simulation.get_active_operation_collection()
        waiting_job_ops_collection = simulation.get_waiting_operation_collection()



