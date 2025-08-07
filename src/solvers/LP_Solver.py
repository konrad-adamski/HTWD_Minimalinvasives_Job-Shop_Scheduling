import time

import pulp
from typing import Literal, Optional
from src.classes.Collection import LiveJobCollection


class Solver:

    @staticmethod
    def build_makespan_problem(
            jobs_collection: LiveJobCollection, var_cat: Literal["Continuous", "Integer"] = "Continuous",
            job_low_bound_on_arrival: bool = True, problem_name:str  = "jss_makespan_problem"):

        # Model initialization and Helper objects ------------------------------------------------------
        problem = pulp.LpProblem(problem_name, pulp.LpMinimize)
        start_times = {}

        machine_names = jobs_collection.get_unique_machine_names()

        # Big-M (Worst-case start difference between operations) ---------------------------------------
        total_duration = jobs_collection.get_total_duration()
        latest_arrival = jobs_collection.get_latest_arrival()
        big_m = latest_arrival + total_duration

        # Create Variables -----------------------------------------------------------------------------
        jobs_collection.sort_operations()
        jobs_collection.sort_jobs_by_arrival()

        for job in jobs_collection.values():
            for operation in job.operations:
                op_numb = operation.position_number

                start_times[(job.id, op_numb)] = pulp.LpVariable(
                    name=f"start_{job.id}_{op_numb}",
                    lowBound=job.arrival if job_low_bound_on_arrival else job.earliest_start,
                    cat=var_cat
                )

        # Operation-level constraints ------------------------------------------------------------------
        for job in jobs_collection.values():
            for operation in job.operations:
                op_numb = operation.position_number

                # Technological constraint: operation order within the job
                prev_op = job.get_previous_operation(op_numb)
                if prev_op:
                    prev_op_numb = prev_op.position_number
                    problem += start_times[(job.id, op_numb)] >= start_times[(job.id, prev_op_numb)] + prev_op.duration

        # Machine-level constraints (NoOverlap) --------------------------------------------------------
        for machine_name in machine_names:
            machine_operations = jobs_collection.get_all_operations_on_machine(machine_name=machine_name)

            for operation_a in machine_operations:
                for operation_b in machine_operations:
                    if operation_a == operation_b:
                        continue

                    job_a = operation_a.job_id
                    job_b = operation_b.job_id
                    op_numb_a = operation_a.position_number
                    op_numb_b = operation_b.position_number
                    duration_a = operation_a.duration
                    duration_b = operation_b.duration

                    y = pulp.LpVariable(
                        name = f"y_{job_a}_{op_numb_a}_{job_b}_{op_numb_b}",
                        cat="Binary"
                    )
                    problem += (start_times[(job_a, op_numb_a)] + duration_a <= start_times[(job_b, op_numb_b)]
                                + big_m * (1 - y))

                    problem += (start_times[(job_b, op_numb_b)] + duration_b <= start_times[(job_a, op_numb_a)]
                                + big_m * y)

        # Makespan -------------------------------------------------------------------------------------
        makespan = pulp.LpVariable("makespan", lowBound=0, cat=var_cat)
        problem += makespan

        for job in jobs_collection.values():
            last_operation = job.get_last_operation()
            if last_operation:
                last_op_number = last_operation.position_number
                last_op_duration = last_operation.duration
                problem += makespan >= start_times[(job.id, last_op_number)] + last_op_duration

        return problem, start_times

    @staticmethod
    def solve_problem(
            problem: pulp.LpProblem, solver_type: Literal["CBC", "HiGHS"] = "CBC",
            print_log_search_progress: bool = False, time_limit: Optional[int] = None,
            relative_gap_limit: float = 0.0, log_file: Optional[str] = None):

        start_timer = time.time()
        solver_args = {
            "gapRel": relative_gap_limit,
            "msg": print_log_search_progress
        }
        if time_limit:
            solver_args["timeLimit"] = time_limit
        if log_file is not None:
            solver_args["logPath"] = log_file

        if solver_type == "HiGHS":
            cmd = pulp.HiGHS_CMD(**solver_args)
        elif solver_type == "CBC":
            cmd = pulp.PULP_CBC_CMD(**solver_args)
        else:
            raise ValueError("solver_type must be either 'CBC' or 'HiGHS'.")
        problem.solve(cmd)
        runtime = round(time.time() - start_timer, 2)
        return runtime

    @classmethod
    def solve_makespan_problem(
            cls, jobs_collection: LiveJobCollection, var_cat: Literal["Continuous", "Integer"] = "Continuous",
            job_low_bound_on_arrival: bool = True, solver_type: Literal["CBC", "HiGHS"] = "CBC",
            print_log_search_progress: bool = False, time_limit: Optional[int] = None,
            relative_gap_limit: float = 0.0, log_file: Optional[str] = None):

        problem, start_times = cls.build_makespan_problem(jobs_collection,var_cat, job_low_bound_on_arrival)
        runtime = cls.solve_problem(
            problem = problem,
            solver_type = solver_type,
            print_log_search_progress = print_log_search_progress,
            time_limit = time_limit,
            relative_gap_limit = relative_gap_limit,
            log_file = log_file
        )

        schedule_job_collection = LiveJobCollection()

        for job in jobs_collection.values():
            for operation in job.operations:
                op_numb = operation.position_number
                start = start_times[(job.id, op_numb)].varValue
                end = start + operation.duration
                schedule_job_collection.add_operation_instance(
                    op=operation,
                    new_start=start,
                    new_end=end
                )

        solver_info = {
            "status": pulp.LpStatus[problem.status],
            "objective_value": pulp.value(problem.objective),
            "num_variables": len(problem.variables()),
            "num_constraints": len(problem.constraints),
            "runtime": runtime
        }
        return schedule_job_collection, solver_info








