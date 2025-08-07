import pulp
from typing import Literal
from src.classes.Collection import LiveJobCollection


class Solver:

    @staticmethod
    def build_makespan_model(
            jobs_collection: LiveJobCollection, var_cat: Literal["Continuous", "Integer"] = "Continuous",
            job_low_bound_on_arrival: bool = True):

        # Model initialization and Helper objects ----------------------------------------------------------------------
        problem = pulp.LpProblem("JSSP_Makespan_Model", pulp.LpMinimize)
        start_times = {}

        machine_names = jobs_collection.get_unique_machine_names()

        # Big-M (Worst-case start difference between operations) -------------------------------------------------------
        total_duration = jobs_collection.get_total_duration()
        latest_arrival = jobs_collection.get_latest_arrival()
        big_m = latest_arrival + total_duration

        # Create Variables ---------------------------------------------------------------------------------------------
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

        # Operation-level constraints ----------------------------------------------------------------------------------
        for job in jobs_collection.values():
            for operation in job.operations:
                op_numb = operation.position_number

                # Technological constraint: operation order within the job
                prev_op = job.get_previous_operation(op_numb)
                if prev_op:
                    prev_op_numb = prev_op.position_number
                    problem += start_times[(job.id, op_numb)] >= start_times[(job.id, prev_op_numb)] + prev_op.duration

        # Machine-level constraints (NoOverlap) ------------------------------------------------------------------------
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

        # Makespan
        makespan = pulp.LpVariable("makespan", lowBound=0, cat=var_cat)
        problem += makespan

        for job in jobs_collection.values():
            last_operation = job.get_last_operation()
            if last_operation:
                last_op_number = last_operation.position_number
                last_op_duration = last_operation.duration
                problem += makespan >= start_times[(job.id, last_op_number)] + last_op_duration





