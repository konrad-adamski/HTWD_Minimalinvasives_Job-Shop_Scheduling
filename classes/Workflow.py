from collections import UserDict
from typing import Optional

import pandas as pd
from pydantic.dataclasses import dataclass


@dataclass
class JobWorkflowOperation:
    job_id: str
    routing_id: Optional[str]
    sequence_number: int
    machine: str
    start_time: Optional[int] = None
    duration: Optional[int] = None
    end_time: Optional[int] = None


class JobOperationWorkflowCollection(UserDict):
    """
    Stores workflow operations per job_id.
    Maps job_id -> List[JobWorkflowOperation] sorted by sequence_number.
    """

    def add_operation(
            self, job_id: str, routing_id: Optional[str], sequence_number: int, machine: str,
            start_time: Optional[int] = None, duration: Optional[int] = None, end_time: Optional[int] = None):
        """
        Adds a new JobWorkflowOperation to the collection.
        All time fields are optional.
        """
        op = JobWorkflowOperation(job_id, routing_id, sequence_number, machine, start_time, duration, end_time)

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

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame,
                       job_column: str = "Job", routing_column: str = "Routing_ID", operation_column: str = "Operation",
                       machine_column = "Machine", start_column: str = "Start",
                       duration_column: str = "Processing Time", end_column: str = "End"):
        """
        Creates a JobOperationWorkflowCollection from a DataFrame.

        :param df: Input DataFrame
        :return: JobOperationWorkflowCollection instance
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
                sequence_number=int(row[operation_column]),
                machine=str(row[machine_column]),
                start_time=row[start_column] if pd.notna(row[start_column]) else None,
                duration=row[duration_column] if pd.notna(row[duration_column]) else None,
                end_time=row[end_column] if pd.notna(row[end_column]) else None
            )

        obj.sort_operations()
        return obj

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


if __name__ == "__main__":

    # Example
    workflow = JobOperationWorkflowCollection()
    workflow.add_operation("Job1", "R1", 0, "M1", 0, 5, 5)
    workflow.add_operation("Job1", "R1", 1, "M2", 6, 3, 9)
    workflow.add_operation("Job2", "R2", 0, "M2", 0, 5, 15)
    workflow.add_operation("Job2", "R2", 1, "M3",  20, 5, 25)

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
