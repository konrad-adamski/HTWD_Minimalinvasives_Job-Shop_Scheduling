from collections import UserDict
from typing import Optional

import pandas as pd
from pydantic.dataclasses import dataclass


@dataclass
class JobWorkflowOperation:
    job_id: str
    routing_id: Optional[str]
    sequence_number: int
    start_time: int
    duration: int
    end_time: int


class JobOperationWorkflowCollection(UserDict):
    """
    Stores workflow operations per job_id.
    Maps job_id -> List[JobWorkflowOperation] sorted by sequence_number.
    """

    def add_operation(self, job_id: str, routing_id: Optional[str], sequence_number: int,
                      start_time: int, duration: int, end_time: int):
        op = JobWorkflowOperation(job_id, routing_id, sequence_number, start_time, duration, end_time)
        self._add(op)

    def _add(self, op: JobWorkflowOperation):
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
            self, job_column: str = "Job", routing_column: str = "Routing_ID",
            operation_column: str = "Operation", start_column: str = "Start",
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
                    start_column: op.start_time,
                    duration_column: op.duration,
                    end_column: op.end_time
                })
        return pd.DataFrame(rows)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame,
                       job_column: str = "Job", routing_column: str = "Routing_ID",
                       operation_column: str = "Operation", start_column: str = "Start",
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

            op = JobWorkflowOperation(
                job_id=str(row[job_column]),
                routing_id=routing_id,
                sequence_number=int(row[operation_column]),
                start_time=int(row[start_column]),
                duration=int(row[duration_column]),
                end_time=int(row[end_column])
            )
            obj._add(op)
        obj.sort_operations()
        return obj


if __name__ == "__main__":

    # Beispiel-Operationen hinzufügen
    workflow = JobOperationWorkflowCollection()
    workflow.add_operation("Job1", "R1", 0, start_time=0, duration=5, end_time=5)
    workflow.add_operation("Job1", "R1", 1, start_time=6, duration=3, end_time=9)
    workflow.add_operation("Job2", "R2", 0, start_time=2, duration=4, end_time=6)
    workflow.add_operation("Job2", "R2", 1, start_time=7, duration=2, end_time=9)

    # Sortieren
    workflow.sort_operations()

    # DataFrame erzeugen und anzeigen
    print("\n--- Workflow als DataFrame ---")
    df = workflow.to_dataframe()
    print(df)

    # Test: Wiederherstellung aus DataFrame
    print("\n--- Wiederherstellung aus DataFrame ---")
    restored = JobOperationWorkflowCollection.from_dataframe(df)
    for job_id, ops in restored.items():
        print(f"Job: {job_id}")
        for op in ops:
            print(f" {op.job_id} Seq {op.sequence_number}, Start {op.start_time}, Duration {op.duration}, End {op.end_time}")
