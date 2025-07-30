from typing import List, Dict
import pandas as pd

from classes.Operation import JobOperation, RoutingOperation, JobOperationView
from classes.Routing import RoutingOperationCollection
from test import JobInformationCollection, JobOperationWorkflowCollection


class JobOperationProblemCollection:
    """
    Manages job-level operations linked to routing definitions.
    """
    def __init__(self, routings_collection: RoutingOperationCollection):
        """
        :param routings_collection: A RoutingOperationCollection reference.
        """
        self.routings_collection = routings_collection
        self.job_operations: List[JobOperation] = []

    def __iter__(self):
        for op in self.job_operations:
            routing_op = self.routings_collection.get_operation(op.routing_id, op.sequence_number)
            if routing_op:
                yield JobOperationView(
                    job_id=op.job_id,
                    routing_id=op.routing_id,
                    sequence_number=op.sequence_number,
                    machine=routing_op.machine,
                    duration=routing_op.duration
                )

    def group_by_job(self) -> Dict[str, List[JobOperationView]]:
        """
        Groups operations by job ID.

        :return: Dictionary with job_id as keys and sorted JobOperationViews as values.
        """
        job_dict: Dict[str, List[JobOperationView]] = {}
        self.sort_problem()
        for op in self:
            if op.job_id not in job_dict:
                job_dict[op.job_id] = []
            job_dict[op.job_id].append(op)
        return job_dict

    def add_job_operation(self, job_id: str, routing_id: str, sequence_number: int):
        """
        Adds a job operation.

        :param job_id: Job identifier
        :param routing_id: Routing identifier
        :param sequence_number: Operation index within the routing
        """
        self.job_operations.append(JobOperation(job_id, routing_id, sequence_number))

    def sort_problem(self):
        """
        Sorts job operations by job ID and sequence number.
        """
        self.job_operations.sort(key=lambda op: (op.job_id, op.sequence_number))

    def to_dataframe(
            self, job_column: str = "Job", routing_column: str = "Routing_ID", operation_column: str = "Operation",
            machine_column: str = "Machine", duration_column: str = "Processing Time") -> pd.DataFrame:
        """
        Converts job operations into a pandas DataFrame.

        :param job_column: Column name for job IDs
        :param routing_column: Column name for routing IDs
        :param operation_column: Column name for sequence numbers
        :param machine_column: Column name for machines
        :param duration_column: Column name for durations
        :return: DataFrame representation of job operations
        :rtype: pd.DataFrame
        """
        rows = []
        for op in self:
            rows.append({
                job_column: op.job_id,
                routing_column: op.routing_id,
                operation_column: op.sequence_number,
                machine_column: op.machine,
                duration_column: op.duration
            })
        return pd.DataFrame(rows)

    @classmethod
    def from_job_dataframe(
            cls, df: pd.DataFrame, routings_collection: RoutingOperationCollection, job_column: str = "Job",
            routing_column: str = "Routing_ID"):
        """
        Creates a JobOperationProblemCollection from a DataFrame with job and routing info.

        :param df: Input DataFrame
        :param routings_collection: RoutingOperationCollection reference
        :param job_column: Column containing job IDs
        :param routing_column: Column containing routing IDs
        :return: Populated JobOperationProblemCollection instance
        :rtype: JobOperationProblemCollection
        """
        # 1. Drop duplikates
        df_clean = df.drop_duplicates(subset=[job_column, routing_column])

        # 2. Create instance
        obj = cls(routings_collection)

        # 3. Process per routing group
        grouped = df_clean.groupby(routing_column)

        for routing_id, group in grouped:
            routing_id = str(routing_id)

            if routing_id not in routings_collection:
                print(f"[Warning] Routing '{routing_id}' doesn't exist – all associated jobs skipped!")
                continue

            for job_id in group[job_column]:
                job_id = str(job_id)
                for op in routings_collection[routing_id]:
                    obj.add_job_operation(job_id, routing_id, op.sequence_number)

        # 4. Sortieren und Rückgabe
        obj.sort_problem()
        return obj

    @classmethod
    def from_collections(
            cls,
            job_info_collection: JobInformationCollection,
            routings_collection: RoutingOperationCollection):
        """
        Creates a JobOperationProblemCollection from job and routing collections.

        :param job_info_collection: Contains job_id → JobInformation (with routing_id)
        :param routings_collection: Routing definitions used to expand operations
        :return: JobOperationProblemCollection instance
        :rtype: JobOperationProblemCollection
        """
        obj = cls(routings_collection)

        for job_id, info in job_info_collection.items():
            routing_id = info.routing_id
            if routing_id is None:
                print(f"[Warning] Job '{job_id}' has no routing_id – skipped.")
                continue

            if routing_id not in routings_collection:
                print(f"[Warning] Routing '{routing_id}' not found – job '{job_id}' skipped.")
                continue

            for op in routings_collection[routing_id]:
                obj.add_job_operation(job_id, routing_id, op.sequence_number)

        obj.sort_problem()
        return obj

    def filter_exclude_operations(
            self,
            excluded_operations: List[tuple[str, int]]):
        """
        Returns a filtered JobOperationProblemCollection excluding specified operations.

        :param excluded_operations: List of (job_id, sequence_number) tuples to exclude
        :return: New filtered JobOperationProblemCollection
        """
        excluded_set = set(excluded_operations)
        filtered = JobOperationProblemCollection(self.routings_collection)

        for op in self.job_operations:
            key = (op.job_id, op.sequence_number)
            if key not in excluded_set:
                filtered.job_operations.append(op)

        return filtered


    def remove_operations_from_workflows(self, workflows: List[JobOperationWorkflowCollection]) -> None:
        """
        Removes in-place all operations that are already present in any of the given workflow collections.

        Operations are matched by (job_id, sequence_number).

        :param workflows: List of JobOperationWorkflowCollection instances
        """
        excluded_keys = set()

        for wflow in workflows:
            for job_id, ops in wflow.items():
                for op in ops:
                    excluded_keys.add((op.job_id, op.sequence_number))

        to_remove = [op for op in self.job_operations if (op.job_id, op.sequence_number) in excluded_keys]
        for op in to_remove:
            self.job_operations.remove(op)

if __name__ == "__main__":
    # 1. Example RoutingOperationCollection
    routings_collection = RoutingOperationCollection()
    routings_collection.add_operation("R1", 0, "M1", 5)
    routings_collection.add_operation("R1", 1, "M2", 3)
    routings_collection.add_operation("R1", 2, "M3", 10)
    routings_collection.add_operation("R2", 0, "M1", 4)
    routings_collection.sort_operations()

    # 2. Example DataFrame Jobs
    df_jobs = pd.DataFrame([
        {"Job": "J25-001", "Routing_ID": "R1"},
        {"Job": "J25-002", "Routing_ID": "R2"},
        {"Job": "J25-003", "Routing_ID": "R1"},
        {"Job": "J25-004", "Routing_ID": "R2"},
        {"Job": "J25-005", "Routing_ID": "R99"}, # absichtlich nicht vorhanden
    ])

    # 3. Create collection
    job_problem = JobOperationProblemCollection.from_job_dataframe(df_jobs, routings_collection)

    print("\n" + "-"*60)
    for op in job_problem:
        print(f"Job {op.job_id}, Operation {op.sequence_number}, Machine {op.machine}, Duration {op.duration}")

    print("\n" + "-" * 60)
    for i, (job_id, ops) in enumerate(job_problem.group_by_job().items()):
        print(f"{i + 1}. Job: {job_id}")
        for op in ops:
            print(f"   Seq {op.sequence_number}, Machine {op.machine}, Duration {op.duration}")







