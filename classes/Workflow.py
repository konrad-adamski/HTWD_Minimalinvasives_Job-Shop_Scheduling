from collections import UserDict
from typing import Optional

import pandas as pd
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class JobWorkflowOperation:
    job_id: str
    routing_id: Optional[str]
    sequence_number: int
    machine: str
    duration: int
    start_time: Optional[int] = None
    end_time: Optional[int] = None


class JobOperationWorkflowCollection(UserDict):
    """
    Stores workflow operations per job_id.
    Maps job_id -> List[JobWorkflowOperation] sorted by sequence_number.
    """

    def add_operation(
            self, job_id: str, routing_id: Optional[str], operation: int, machine: str,
            duration: int, start: Optional[int] = None, end: Optional[int] = None):
        """
        Adds a new JobWorkflowOperation to the collection.
        All time fields are optional.
        """
        op = JobWorkflowOperation(
            job_id=job_id,
            routing_id=routing_id,
            sequence_number= operation,
            machine=machine,
            duration= duration,
            start_time = start,
            end_time = end
        )

        if job_id not in self:
            self[job_id] = []
        self[job_id].append(op)


    def remove_operation(self, job_id: str, sequence_number: int):
        """
        Removes the first operation with the given sequence_number for the specified job_id.
        Deletes the job entry if no operations remain.
        """
        if job_id in self:
            for op in self[job_id]:
                if op.sequence_number == sequence_number:
                    self[job_id].remove(op)
                    break

            if not self[job_id]:
                del self[job_id]

    def add_operation_instance(self, op: JobWorkflowOperation):
        """
        Adds a complete JobWorkflowOperation instance to the collection.
        """
        if op.job_id not in self:
            self[op.job_id] = []
        self[op.job_id].append(op)

    def sort_operations(self):
        """
        Sorts operations for each job by sequence_number.
        """
        for job_id in self:
            self[job_id].sort(key=lambda op: op.sequence_number)

    def to_dataframe(
            self, job_column: str = "Job", routing_column: str = "Routing_ID", operation_column: str = "Operation",
            machine_column: str = "Machine", start_column: str = "Start",
            duration_column: str = "Processing Time", end_column: str = "End") -> pd.DataFrame:
        """
        Converts the workflow operations to a pandas DataFrame.

        :return: DataFrame with given columns
        """
        rows = []
        for job_id, ops in self.items():
            for op in ops:
                rows.append({
                    job_column: op.job_id,
                    routing_column: op.routing_id,
                    operation_column: op.sequence_number,
                    machine_column: op.machine,
                    start_column: op.start_time,
                    duration_column: op.duration,
                    end_column: op.end_time
                })
        return pd.DataFrame(rows)


    def get_operation(self, job_id: str, sequence_number: int) -> Optional[JobWorkflowOperation]:
        """
        Gibt die erste Operation mit dem gegebenen sequence_number für job_id zurück.
        """
        if job_id in self:
            for op in self[job_id]:
                if op.sequence_number == sequence_number:
                    return op
        return None

    def get_unique_machines(self) -> set:
        """
        Gibt die Menge aller in den Operationen verwendeten Maschinen zurück.
        """
        machines = {
            op.machine
            for ops in self.values()
            for op in ops
            if op.machine is not None
        }
        return machines

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame,
                       job_column: str = "Job", routing_column: str = "Routing_ID", operation_column: str = "Operation",
                       machine_column="Machine", start_column: str = "Start",
                       duration_column: str = "Processing Time", end_column: str = "End"):
        """
        Creates a JobOperationWorkflowCollection from a DataFrame.

        :param df: Input DataFrame with one row per operation.
        :param job_column: str – Column with job ID (e.g., "Job").
        :param routing_column: str – Column with routing ID (optional, can be NaN).
        :param operation_column: str – Column with operation number (int).
        :param machine_column: str – Column with machine name (str).
        :param start_column: str – Column with start time (optional, int or NaN).
        :param duration_column: str – Column with processing time (int, required).
        :param end_column: str – Column with end time (optional, int or NaN).
        :return: JobOperationWorkflowCollection with loaded operations.
        """
        obj = cls()
        if df is None or df.empty:
            return obj

        has_routings = routing_column in df.columns

        for _, row in df.iterrows():
            routing_id = str(row[routing_column]) if has_routings and pd.notna(row[routing_column]) else None

            obj.add_operation(
                job_id=str(row[job_column]),
                routing_id=routing_id,
                operation=int(row[operation_column]),
                machine=str(row[machine_column]),
                start=row[start_column] if pd.notna(row[start_column]) else None,
                duration=row[duration_column],
                end=row[end_column] if pd.notna(row[end_column]) else None
            )

        obj.sort_operations()
        return obj


    @classmethod
    def subtract_by_job_operation_collections(
            cls, main: "JobOperationWorkflowCollection",
            *exclude_collections: "JobOperationWorkflowCollection") -> "JobOperationWorkflowCollection":
        """
        Returns a new collection (of the same class) containing only those operations from 'main'
        whose (job_id, sequence_number) identifiers do not appear in any of the exclusion collections.

        This method removes only matching operations by (job_id, sequence_number), not by full object equality.

        :param main: The base collection containing all operations to filter.
        :param exclude_collections: Any number of exclusion collections. Operations with the same
                                    (job_id, sequence_number) will be removed from the result.
        :return: A new JobOperationWorkflowCollection with filtered operations.
        """
        result = cls()

        # Set of (job_id, sequence_number) tuples to exclude
        excluded_pairs = set()
        for collection in exclude_collections:
            for ops in collection.values():
                for op in ops:
                    excluded_pairs.add((op.job_id, op.sequence_number))

        # Keep only those operations not matching any (job_id, sequence_number)
        for ops in main.values():
            for op in ops:
                if (op.job_id, op.sequence_number) not in excluded_pairs:
                    result.add_operation_instance(op)
                    print(op.job_id, op.sequence_number)

        return result


if __name__ == "__main__":

    # Example
    workflow = JobOperationWorkflowCollection()
    workflow.add_operation("Job1", "R1", 0, "M1", start = 0, duration = 5, end = 5)
    workflow.add_operation("Job1", "R1", 1, "M2", start = 6, duration = 3, end = 9)
    workflow.add_operation("Job2", "R2", 0, "M2", start = 2, duration = 3, end = 5)
    workflow.add_operation("Job2", "R2", 1, "M3",  start = 6, duration = 4, end = 10)

    # Sortieren
    workflow.sort_operations()

    # Create DataFrame
    print("\n--- Workflow as DataFrame ---")
    df = workflow.to_dataframe()
    print(df)

    # Test: from DataFrame
    workflow_restored = JobOperationWorkflowCollection.from_dataframe(df)
    for job_id, ops in workflow_restored.items():
        print(f"Job: {job_id}")
        for op in ops:
            print(f" {op.job_id} Seq {op.sequence_number}, Start {op.start_time}, Duration {op.duration}, End {op.end_time}")
