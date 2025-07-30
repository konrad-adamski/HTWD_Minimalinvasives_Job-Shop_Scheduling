from dataclasses import dataclass, astuple
from collections import UserDict
from typing import Optional

import pandas as pd

@dataclass
class JobInformation:
    earliest_start: int
    deadline: int


class JobInformationCollection(UserDict):
    def add_job(self, job_id: str, earliest_start: int, deadline: int):
        self[job_id] = JobInformation(earliest_start, deadline)

    def get_earliest_start(self, job_id: str) -> int:
        return self[job_id].earliest_start

    def get_deadline(self, job_id: str) -> int:
        return self[job_id].deadline

    def get_jobs_by_earliest_start(self, time_point: int):
        return {
            job_id: info
            for job_id, info in self.items()
            if info.earliest_start == time_point
        }

    def get_dict(self) -> dict:
        return {
            job_id: astuple(info)
            for job_id, info in self.items()
        }

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        job_column: str = "Job",
        earliest_start_column: str = "Ready Time",
        deadline_column: str = "Deadline"
    ):
        obj = cls()
        for _, row in df.iterrows():
            job_id = row[job_column]
            earliest_start = int(row[earliest_start_column])
            deadline = int(row[deadline_column])
            obj.add_job(job_id, earliest_start, deadline)
        return obj


@dataclass
class RoutingOperation:
    sequence_number: int
    machine: str
    duration: int


@dataclass
class JobOperation:
    sequence_number: int
    machine: str
    duration: int
    start_time: Optional[int] = None
    end_time: Optional[int] = None



class JobOperationCollection(UserDict):
    def add_operation(self, job_id: str, operation_sequence_number: int, machine: str, duration: int):
        if job_id not in self:
            self[job_id] = []
        self[job_id].append(JobOperation(operation_sequence_number, machine, duration))

    def get_operations(self, job_id: str) -> list[JobOperation]:
        return self[job_id]

    def get_problem_dict(self) -> dict:
        return {
            job_id: [(op.sequence_number, op.machine, op.duration) for op in operations]
            for job_id, operations in self.items()
        }

    def get_workflow_dict(self) -> dict:
        return {
            job_id: [astuple(op) for op in operations]
            for job_id, operations in self.items()
        }

    @classmethod
    def _from_dataframe(
            cls, df: pd.DataFrame, job_column: str, operation_column: str,
            machine_column: str, duration_column: str,
            start_column: Optional[str] = None, end_column: Optional[str] = None):

        obj = cls()
        for _, row in df.iterrows():
            job_id = row[job_column]
            operation_number = int(row[operation_column])
            machine = str(row[machine_column])
            duration = int(row[duration_column])

            start_time = int(row[start_column]) if start_column and pd.notna(row[start_column]) else None
            end_time = int(row[end_column]) if end_column and pd.notna(row[end_column]) else None

            operation = JobOperation(
                sequence_number=operation_number,
                machine=machine,
                duration=duration,
                start_time=start_time,
                end_time=end_time
            )

            if job_id not in obj:
                obj[job_id] = []
            obj[job_id].append(operation)

        # Optional: sortiere nach Operation-ID
        for job_id in obj:
            obj[job_id].sort(key=lambda op: op.sequence_number)

        return obj

    @classmethod
    def from_problem_dataframe(
            cls, df: pd.DataFrame, job_column: str = "Job", operation_column: str = "Operation",
            machine_column: str = "Machine", duration_column: str = "Processing Time"):
        return cls._from_dataframe(
            df,
            job_column=job_column,
            operation_column=operation_column,
            machine_column=machine_column,
            duration_column=duration_column,
            start_column=None,
            end_column=None
        )

    @classmethod
    def from_workflow_dataframe(
            cls, df: pd.DataFrame, job_column: str = "Job", operation_column: str = "Operation",
            machine_column: str = "Machine", duration_column: str = "Processing Time",
            start_column: str = "Start", end_column: str = "End"):

        return cls._from_dataframe(
            df,
            job_column=job_column,
            operation_column=operation_column,
            machine_column=machine_column,
            duration_column=duration_column,
            start_column=start_column,
            end_column=end_column
        )
