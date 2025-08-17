from decimal import Decimal
from typing import Optional

from config.project_config import get_solver_logs_path
from src.EmailNotifier import EmailNotifier
from src.Logger import Logger
from src.domain.Collection import LiveJobCollection
from src.domain.Query import JobQuery, ExperimentQuery, MachineQuery
from src.domain.orm_models import Experiment
from src.simulation.ProductionSimulation import ProductionSimulation
from src.solvers.CP_Solver import Solver

email_notifier = EmailNotifier()

def run_experiment(experiment_id: int,  shift_length: int, total_shift_number: int, logger: Logger):
    experiment = ExperimentQuery.get_experiment(experiment_id)

    source_name = experiment.routing_source.name
    max_bottleneck_utilization = experiment.max_bottleneck_utilization

    w_t, w_e, w_dev = experiment.get_solver_weights()

    # Preparation  ----------------------------------------------------------------------------------
    simulation = ProductionSimulation(sigma=experiment.sim_sigma, verbose=False)

    # Jobs Collection
    jobs = JobQuery.get_by_source_name_max_util_and_lt_arrival(
        source_name=source_name,
        max_bottleneck_utilization=Decimal(f"{max_bottleneck_utilization}"),
        arrival_limit=60 * 24 * total_shift_number
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

    # Collections(empty)
    schedule_jobs_collection = LiveJobCollection()  # pseudo previous schedule
    active_job_ops_collection = LiveJobCollection()

    waiting_job_ops_collection = LiveJobCollection()

    # Shifts ----------------------------------------------------------------------------------------
    for shift_number in range(1, total_shift_number + 1):
        shift_start = shift_number * shift_length
        shift_end = (shift_number + 1) * shift_length
        logger.info(f"Experiment {experiment_id} shift {shift_number}: {shift_start} to {shift_end}")

        new_jobs_collection = jobs_collection.get_subset_by_earliest_start(earliest_start=shift_start)
        current_jobs_collection = new_jobs_collection + waiting_job_ops_collection

        # Scheduling --------------------------------------------------------------
        solver = Solver(
            jobs_collection=current_jobs_collection,
            logger = logger,
            schedule_start=shift_start
        )

        solver.build_model__absolute_lateness__start_deviation__minimization(
            previous_schedule_jobs_collection=schedule_jobs_collection,
            active_jobs_collection=active_job_ops_collection,
            w_t=w_t, w_e=w_e, w_dev=w_dev
        )

        solver.log_model_info()

        file_path = get_solver_logs_path(
            sub_directory=f"Experiment_{experiment_id:03d}",
            file_name=f"Shift_{shift_number:02d}.log",
            as_string=True
        )

        solver.solve_model(
            gap_limit=0.005,
            time_limit=60*30,
            log_file=file_path,
            bound_relative_change= 0.01,
            bound_no_improvement_time= 60*4,
            bound_warmup_time=60*1,
        )

        solver.log_solver_info()
        schedule_jobs_collection = solver.get_schedule()

        ExperimentQuery.save_schedule_jobs(
            experiment_id=experiment_id,
            shift_number=shift_number,
            live_jobs=schedule_jobs_collection.values(),
        )

        # Simulation --------------------------------------------------------------
        simulation.run(
            schedule_collection=schedule_jobs_collection,
            start_time=shift_start,
            end_time=shift_end
        )

        active_job_ops_collection = simulation.get_active_operation_collection()
        waiting_job_ops_collection = simulation.get_waiting_operation_collection()

        if shift_number % 3 == 0:
            notify(experiment, logger, shift_number)

    # Save entire Simulation -------------------------------------------------------
    entire_simulation_jobs = simulation.get_entire_finished_operation_collection()
    ExperimentQuery.save_simulation_jobs(
        experiment_id=experiment_id,
        live_jobs=entire_simulation_jobs.values(),
    )
    logger.info(f"Experiment {experiment_id} finished")
    notify(experiment, logger)



def notify(experiment:Experiment, logger: Logger, shift_number: Optional[int] = None):
    experiment_info = f"Experiment {experiment.id} "
    if shift_number:
        experiment_info += (f"Shift {shift_number} - "
                            + f"Absolute Lateness ratio: {experiment.absolute_lateness_ratio}, "
                            + f"Inner Tardiness ratio: {experiment.inner_tardiness_ratio}, "
                            + f"Max bottleneck utilization: {experiment.max_bottleneck_utilization}, "
                            + f"Simulation sigma: {experiment.sim_sigma}")
        last_lines = 70
    else:
        experiment_info += "finished"
        last_lines = 3

    email_notifier.send_log_tail(
        subject=f"{experiment_info}",
        log_file= logger.get_log_file_path(),
        lines = last_lines
    )




