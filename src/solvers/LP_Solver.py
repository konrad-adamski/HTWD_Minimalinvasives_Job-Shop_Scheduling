import math
import time

import pulp
from typing import Literal, Optional
from src.domain.Collection import LiveJobCollection

class Solver:

    def __init__(
            self, jobs_collection: LiveJobCollection, problem_name:str  = "jss_makespan_problem",
            var_cat: Literal["Continuous", "Integer"] = "Continuous", epsilon: float = 0.2):

        self.jobs_collection = jobs_collection
        self.runtime = None
        self.var_cat = var_cat
        # Model initialization and Helper objects ------------------------------------------------------
        self.problem = pulp.LpProblem(problem_name, pulp.LpMinimize)
        self.start_times = {}

        self.machine_names = jobs_collection.get_unique_machine_names()

        # Big-M (Worst-case start difference between operation starts) ---------------------------------
        total_duration = jobs_collection.get_total_duration()
        latest_earliest_start = jobs_collection.get_latest_earliest_start()
        self.big_m = latest_earliest_start + total_duration

        # Create Variables -----------------------------------------------------------------------------
        self.jobs_collection.sort_operations()
        self.jobs_collection.sort_jobs_by_arrival()

        for job in self.jobs_collection.values():
            for operation in job.operations:
                op_numb = operation.position_number

                self.start_times[(job.id, op_numb)] = pulp.LpVariable(
                    name=f"start_{job.id}_{op_numb}",
                    lowBound=job.earliest_start,
                    cat=self.var_cat
                )

        # Operation-level constraints ------------------------------------------------------------------
        for job in self.jobs_collection.values():
            for operation in job.operations:
                op_numb = operation.position_number

                # Technological constraint: operation order within the job
                prev_op = job.get_previous_operation(op_numb)
                if prev_op:
                    prev_op_numb = prev_op.position_number
                    self.problem += (self.start_times[(job.id, op_numb)]
                                     >= self.start_times[(job.id, prev_op_numb)] + prev_op.duration)

        # Machine-level constraints (NoOverlap) --------------------------------------------------------
        for machine_name in self.machine_names:
            machine_operations = self.jobs_collection.get_all_operations_on_machine(machine_name=machine_name)

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
                        name=f"y_{job_a}_{op_numb_a}_{job_b}_{op_numb_b}",
                        cat="Binary"
                    )
                    self.problem += (self.start_times[(job_a, op_numb_a)] + duration_a + epsilon
                                     <= self.start_times[(job_b, op_numb_b)] + self.big_m * (1 - y))

                    self.problem += (self.start_times[(job_b, op_numb_b)] + duration_b + epsilon
                                     <= self.start_times[(job_a, op_numb_a)] + self.big_m * y)


    def build_makespan_problem(self):
        makespan = pulp.LpVariable("makespan", lowBound=0, cat=self.var_cat)
        self.problem += makespan

        for job in self.jobs_collection.values():
            last_operation = job.get_last_operation()
            if last_operation:
                last_op_number = last_operation.position_number
                last_op_duration = last_operation.duration
                self.problem += makespan >= self.start_times[(job.id, last_op_number)] + last_op_duration


    def solve_problem(
            self, solver_type: Literal["CBC", "HiGHS"] = "CBC",
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
        self.problem.solve(cmd)
        self.runtime = round(time.time() - start_timer, 2)


    def get_schedule(self):
        if self.runtime:
            schedule_job_collection = LiveJobCollection()

            for job in self.jobs_collection.values():
                for operation in job.operations:
                    op_numb = operation.position_number
                    start = round(self.start_times[(job.id, op_numb)].varValue, 1)
                    end = start + operation.duration
                    schedule_job_collection.add_operation_instance(
                        op=operation,
                        new_start=start,
                        new_end=end
                    )
            return schedule_job_collection
        else:
            return "Solver not finished (or even not started)."

    def get_solver_info(self):
        solver_info = {
            "status": pulp.LpStatus[self.problem.status],
            "objective_value": pulp.value(self.problem.objective),
            "num_variables": len(self.problem.variables()),
            "num_constraints": len(self.problem.constraints),
            "runtime": self.runtime
        }
        return solver_info










