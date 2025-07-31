from dataclasses import replace
from typing import List, Optional, Dict, Tuple

from classes.JobInformation import JobInformationCollection
from classes.Workflow import JobOperationWorkflowCollection, JobWorkflowOperation
from src.simulation.sim_utils import duration_log_normal,get_duration, get_time_str
from src.simulation.Machine import Machine
import time
import simpy
import pandas as pd


class ProductionSimulation:
    def __init__(
            self, job_information_collection: Optional[JobInformationCollection] = None,
            shift_length: int = 1440, sigma: float = 0.2, verbose: bool = True):

        if job_information_collection is not None:
            self.job_information_collection = job_information_collection
        else:
            self.job_information_collection = JobInformationCollection()
        self.verbose = verbose
        self.sigma = sigma
        self.shift_length = shift_length
        self.machines = {}
        self.start_time = 0
        self.pause_time = 0

        self.active_operations: Dict[Tuple[str, int], JobWorkflowOperation] = {}  # (job_id, operation) → operation
        self.finished_operations_collection = JobOperationWorkflowCollection()
        self.entire_finished_operations_collection = JobOperationWorkflowCollection()

        self.controller = None
        self.env = None

    def reload_machines(self):                                  # for new SimPy environment (continue)
        for name, old_machine in self.machines.items():
            self.machines[name] = Machine(self.env, name)

    def add_new_machines(self, machines: set):
        for m in machines:
            if m not in self.machines:
                self.machines[m] = Machine(self.env, m)

    def job_process(self, job_id: str, job_operations: List[JobWorkflowOperation]):

        if job_id in self.job_information_collection:           # for FCFS
            job_info = self.job_information_collection[job_id]
            delay = max(job_info.earliest_start - self.env.now, 0)
            yield self.env.timeout(delay)
        for op in job_operations:
            machine = self.machines[op.machine]
            planned_start = op.start_time if op.start_time is not None else self.start_time
            delay = max(planned_start - self.env.now, 0)
            yield self.env.timeout(delay)


            with machine.request() as req:
                yield req
                sim_start = self.env.now
                self.log_job_started_on_machine(sim_start, job_id, machine)

                sim_duration = duration_log_normal(op.duration, sigma=self.sigma)
                self.register_active_operation(job_op=op, sim_start=sim_start, sim_duration=sim_duration)

                yield self.env.timeout(sim_duration)
                sim_end = self.env.now
                self.log_job_finished_on_machine(sim_end, job_id, machine, sim_duration)

            self.add_finished_operation(job_op=op, sim_start=sim_start, sim_end=sim_end)

    def resume_operation_process(self, job_op: JobWorkflowOperation):
        remaining_time = max(0, job_op.end_time - self.start_time)

        machine = self.machines[job_op.machine]
        self.log_job_resumed_on_machine(time_stamp=self.env.now, remaining_time=remaining_time, job_op=job_op)

        with machine.request() as req:
            yield req
            yield self.env.timeout(remaining_time)
            sim_end = self.env.now
            self.log_job_finished_on_machine(sim_end, job_op.job_id, machine, remaining_time)
        self.add_finished_operation(job_op=job_op,sim_start=job_op.start_time,sim_end=sim_end,)

    def run(
            self, schedule_collection: Optional[JobOperationWorkflowCollection] = None,
            start_time: int = 0, end_time: int | None = None):
        self.start_time = start_time
        self.pause_time = end_time
        self.env = simpy.Environment(initial_time=start_time)
        self.reload_machines()
        self.finished_operations_collection = JobOperationWorkflowCollection()

        for job_op in self.active_operations.values():
            self.env.process(self.resume_operation_process(job_op))

        if schedule_collection is not None:
            machines = schedule_collection.get_unique_machines()
            self.add_new_machines(machines)

            for job_id, operations in schedule_collection.items():
                self.env.process(self.job_process(job_id, operations))

        if end_time is not None:
            self.env.run(until=end_time)
        else:
            self.env.run()

    def initialize_run(self, schedule_collection: JobOperationWorkflowCollection, start_time: int = 0):
        end_time = start_time + self.shift_length
        self.run(schedule_collection=schedule_collection, start_time=start_time, end_time=end_time)

    def continue_run(self, schedule_collection: Optional[JobOperationWorkflowCollection] = None):
        if self.pause_time is None:
            raise ValueError("Simulation must be initialized before continuing.")

        start_time = self.pause_time
        end_time = start_time + self.shift_length
        self.run(schedule_collection=schedule_collection, start_time=start_time, end_time=end_time)

    def log_job_started_on_machine(self, time_stamp, job_id, machine):
        if self.verbose:
            print(f"[{get_time_str(time_stamp)}] Job {job_id} started on {machine.name}")
        if self.controller:
            self.controller.job_started_on_machine(time_stamp, job_id, machine)
            time.sleep(0.05)

    def log_job_finished_on_machine(self, time_stamp, job_id, machine, sim_duration):
        if self.verbose:
            print(f"[{get_time_str(time_stamp)}] Job {job_id} finished on {machine.name} "
                  + f"(after {get_duration(sim_duration)})")
        if self.controller:
            self.controller.job_finished_on_machine(time_stamp, job_id, machine, sim_duration)
            time.sleep(0.14)

    def log_job_resumed_on_machine(self, time_stamp, remaining_time, job_op: JobWorkflowOperation):
        if self.verbose:
            print(f"[{get_time_str(time_stamp)}] Job {job_op.job_id}, Operation {job_op.sequence_number} "
              + f"resumed on {job_op.machine} with {get_duration(remaining_time)} left)")
        if self.controller:
            self.controller.resume()
            time.sleep(0.14)

    def register_active_operation(self, job_op: JobWorkflowOperation, sim_start, sim_duration):
        updated_op = replace(
            job_op,
            duration=sim_duration,
            start_time=sim_start,
            end_time=sim_start + sim_duration
        )
        self.active_operations[(job_op.job_id, job_op.sequence_number)] = updated_op


    def add_finished_operation(self, job_op: JobWorkflowOperation, sim_start, sim_end):
        updated_op = replace(
            job_op,
            duration=sim_end - sim_start,
            start_time=sim_start,
            end_time=sim_end
        )
        self.finished_operations_collection.add_operation_instance(updated_op)
        self.entire_finished_operations_collection.add_operation_instance(updated_op)

        self.active_operations.pop((job_op.job_id, job_op.sequence_number), None)

    def get_active_operations(self) -> JobOperationWorkflowCollection:
        collection = JobOperationWorkflowCollection()
        for op in self.active_operations.values():
            collection.add_operation_instance(op)
        return collection

    def set_active_operations(self, active_operations_collection: JobOperationWorkflowCollection):
        self.active_operations = {}
        for job_id, op_list in active_operations_collection.items():
            for op in op_list:
                key = (op.job_id, op.sequence_number)
                self.active_operations[key] = op

    def get_finished_operations(self) -> JobOperationWorkflowCollection:
        return self.finished_operations_collection


if __name__ == "__main__":

    from configs.path_manager import get_path

    basic_data_path = get_path("data", "examples")
    df_schedule = pd.read_csv(basic_data_path / "lateness_schedule_day_01.csv")

    print("\n", "---" * 20, "Schedule", "---" * 20)
    schedule_collection = JobOperationWorkflowCollection.from_dataframe(df_schedule)

    print("\n", "---" * 20, "Simulation", "---" * 20)
    simulation = ProductionSimulation(shift_length=1440, sigma= 0.02)

    simulation.run(schedule_collection, start_time= 1440, end_time= 2800)

    print("\n","---" * 20, "Finished Operations", "---" * 20)
    finished_operations = simulation.get_finished_operations()

    df_finished = finished_operations.to_dataframe()
    print(df_finished.head(5))

    print("\n", "---" * 20, "Active Operations", "---" * 20)
    active_operations = simulation.get_active_operations()

    df_active = active_operations.to_dataframe()
    print(df_active.head(5))

    print("\n", "---" * 20, "Waiting Operations", "---" * 20)

    waiting_operation = JobOperationWorkflowCollection.subtract_by_job_operation_collections(
        main = schedule_collection,
        exclude1=finished_operations,
        exclude2=active_operations
    )
    df_waiting = waiting_operation.to_dataframe()
    print(df_waiting.head(5))

    print("---"*60)
    print(f"\nScheduleOperations count: {len(df_schedule)}")
    print(f"Finished Operations count: {len(df_finished)}")
    print(f"Active operations count: {len(active_operations)}")
    print(f"Waiting operations count: {len(waiting_operation)}")