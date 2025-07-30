from collections import UserDict
from typing import List, Optional

import pandas as pd
from pydantic.dataclasses import dataclass


@dataclass
class JobInformation:
    arrival_time: int
    earliest_start: int
    deadline: int
    routing_id: Optional[str]

class JobInformationCollection(UserDict):
    """
    Stores arrivals, earliest start times and deadlines per job.
    """
    def add_job(
            self, job_id: str, arrival_time: int, earliest_start: int,
            deadline: int, routing_id: Optional[str] = None):
        """
        Adds or updates job timing information and optional routing reference.

        :param job_id: ID of the job
        :param earliest_start: Earliest possible start time
        :param deadline: Desired deadline
        :param routing_id: Optional routing ID
        :param arrival_time: Optional arrival time (default: 0)
        """
        self[job_id] = JobInformation(
            arrival_time=arrival_time,
            earliest_start=earliest_start,
            deadline=deadline,
            routing_id=routing_id,
        )

    def to_dataframe(
            self, job_column: str = "Job", routing_column: str = "Routing_ID",
            arrival_column: str = "Arrival", earliest_start_column: str = "Ready Time",
            deadline_column: str = "Deadline") -> pd.DataFrame:
        """
        Converts the collection to a DataFrame.

        :param job_column: Name of the job ID column
        :param routing_column: Name of the routing ID column
        :param arrival_column: Name of the arrival time column
        :param earliest_start_column: Name of the earliest start time column
        :param deadline_column: Name of the deadline column
        :return: DataFrame with job timing and optional routing info
        """
        rows = []
        for job_id, info in self.items():
            rows.append({
                job_column: job_id,
                routing_column: info.routing_id,
                arrival_column: info.arrival_time,
                earliest_start_column: info.earliest_start,
                deadline_column: info.deadline
            })
        return pd.DataFrame(rows)

    @classmethod
    def from_dataframe(
            cls, df: pd.DataFrame, job_column: str = "Job", routing_column: str = "Routing_ID",
            arrival_column: str = "Arrival", earliest_start_column: str = "Ready Time",
            deadline_column: str = "Deadline"):
        """
        Creates a JobInformationCollection from a DataFrame.
        Removes duplicate jobs (keeps the first occurrence).
        """
        df_clean = df.drop_duplicates(subset=[job_column], keep="first")

        obj = cls()
        for _, row in df_clean.iterrows():
            obj.add_job(
                job_id=str(row[job_column]),
                arrival_time=int(row[arrival_column]),
                earliest_start=int(row[earliest_start_column]),
                deadline=int(row[deadline_column]),
                routing_id=str(row[routing_column]) if routing_column in row and pd.notna(row[routing_column]) else None
            )
        return obj

    def get_subset(self, earliest_start: int, planable_job_ids: Optional[List[str]] = None):
        """
        Returns a subset with jobs that are either newly arrived or have unscheduled operations.

        :param earliest_start: Current time (selects newly arrived jobs)
        :param planable_job_ids: Jobs with pending operations
        :return: Filtered JobInformationCollection
        """
        subset = JobInformationCollection()
        for job_id, info in self.items():
            if info.earliest_start == earliest_start or (planable_job_ids and job_id in planable_job_ids):
                subset[job_id] = info
        return subset


if __name__ == "__main__":

    # Example
    df_jobs = pd.DataFrame([
        {"Job": "J1", "Arrival": 10, "Ready Time": 1440, "Deadline": 5000, "Routing_ID": "R10"},
        {"Job": "J2", "Arrival": 10, "Ready Time": 1440, "Deadline": 5000},
        {"Job": "J1", "Arrival": 10, "Ready Time": 1440, "Deadline": 4000},
        {"Job": "J3", "Arrival": 1600, "Ready Time": 2880, "Deadline": 4000}
    ])

    job_collection = JobInformationCollection.from_dataframe(df_jobs)

    for job_id, info in job_collection.items():
        print(f"Job {job_id}: Start={info.earliest_start}, Deadline={info.deadline}")
    print("-"*60)
    for job_id, info in job_collection.get_subset(earliest_start=2880, planable_job_ids=["J2"]).items():
        print(f"{job_id =} {info.earliest_start = } {info.deadline =}")

