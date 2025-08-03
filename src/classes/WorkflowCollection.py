from collections import UserDict
from dataclasses import dataclass
from typing import List, Union, Optional

from src.classes.orm_models import ScheduleOperation, JobOperation


@dataclass
class JobOperationCollection(UserDict):
    """
    Kapselt eine Sammlung von JobWorkflowOperationen, gruppiert nach Job-ID.
    """
    def __init__(self):
        super().__init__()

    def _add_from_operation(self, operation: Union[ScheduleOperation, JobOperation]):
        job_id = operation.job_id

        job_op = JobOperation(
            job=operation.job,
            routing_id=operation.job.routing_id,
            position_number=operation.position_number,
            machine=operation.machine,
            start=operation.start,
            duration=operation.duration,
            end=operation.end
        )

        if job_id not in self.data:
            self.data[job_id] = []
        self.data[job_id].append(job_op)

    def add_from_operation_list(self, operations: List[Union[ScheduleOperation, JobOperation]]):
        for op in operations:
            self._add_from_operation(op)



    def add_operation_instance(self, op: JobOperation):
        """
        Adds a complete JobWorkflowOperation instance to the collection.
        """
        if op.job_id not in self:
            self[op.job_id] = []
        self[op.job_id].append(op)

    def sort_operations(self):
        """
        Sorts operations for each job by position_number.
        """
        for job_id in self:
            self[job_id].sort(key=lambda op: op.position_number)


    def get_operations(self, job_id: str) -> List[JobOperation]:
        return self.data.get(job_id, [])

    def all_operations(self) -> List[JobOperation]:
        """
        Gibt alle WorkflowOperationen als flache Liste zurück.
        """
        return [op for ops in self.data.values() for op in ops]

    def add_operation(
            self, job_id: str, routing_id: Optional[str], experiment_id: Optional[int], operation: int, machine: str,
            duration: int, start: Optional[int] = None, end: Optional[int] = None):
        """
        Adds a new JobWorkflowOperation to the collection.
        All time fields are optional.
        """
        op = JobOperation(
            job_id=job_id,
            routing_id=routing_id,
            experiment_id=experiment_id,
            position_number=operation,
            machine=machine,
            duration=duration,
            start_time=start,
            end_time=end,
        )

        if job_id not in self:
            self[job_id] = []
        self[job_id].append(op)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame,
                       job_column: str = "Job", routing_column: str = "Routing_ID", operation_column: str = "Operation",
                       machine_column="Machine", start_column: str = "Start",
                       duration_column: str = "Processing Time",
                       end_column: str = "End", experiment_id: int = 1) -> JobOperationWorkflowCollection:
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
        :param experiment_id: Experiment ID (optional, int).
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
                experiment_id=experiment_id,
                routing_id=routing_id,
                operation=int(row[operation_column]),
                machine=str(row[machine_column]),
                start=row[start_column] if pd.notna(row[start_column]) else None,
                duration=row[duration_column],
                end=row[end_column] if pd.notna(row[end_column]) else None
            )

        obj.sort_operations()
        return obj

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
                    operation_column: op.position_number,
                    machine_column: op.machine,
                    start_column: op.start_time,
                    duration_column: op.duration,
                    end_column: op.end_time
                })
        return pd.DataFrame(rows)

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
    def _subtract_by_job_operation_collection(
        cls,
        main: JobOperationWorkflowCollection,
        exclude: JobOperationWorkflowCollection
    ) -> JobOperationWorkflowCollection:
        """
        Gibt eine neue Collection zurück, die nur jene Operationen aus 'main' enthält,
        deren (job_id, position_number) nicht in der 'exclude'-Collection vorkommen.

        :param main: Hauptcollection mit allen Operationen
        :param exclude: Collection mit auszuschließenden Operationen
        :return: Neue Collection mit gefilterten Operationen
        """
        result = cls()

        excluded_pairs = {
            (op.job_id, op.position_number)
            for ops in exclude.values()
            for op in ops
        }

        for ops in main.values():
            for op in ops:
                if (op.job_id, op.position_number) not in excluded_pairs:
                    result.add_operation_instance(op)

        return result

    def __truediv__(self, other: JobOperationWorkflowCollection) -> JobOperationWorkflowCollection:
        """
        Überlädt den `/`-Operator, um eine Collection von Operationen zu erzeugen,
        bei der alle (job_id, position_number) aus 'other' entfernt wurden.

        :param other: Collection mit auszuschließenden Operationen
        :return: Neue gefilterte Collection
        """
        return self.__class__._subtract_by_job_operation_collection(main=self, exclude=other)
