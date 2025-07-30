from collections import UserDict
from dataclasses import astuple

import pandas as pd
from pydantic.dataclasses import dataclass


@dataclass
class JobInformation:
    earliest_start: int
    deadline: int



class JobInformationCollection(UserDict):
    """
    Stores earliest start times and deadlines per job.
    """

    def add_job(self, job_id: str, earliest_start: int, deadline: int):
        """
        Adds or updates job timing information.

        :param job_id: ID of the job
        :param earliest_start: Earliest possible start time
        :param deadline: Desired deadline
        """
        self[job_id] = JobInformation(earliest_start, deadline)

    def to_dataframe(self, job_column: str = "Job", start_column: str = "Ready Time",
                     deadline_column: str = "Deadline") -> pd.DataFrame:
        """
        Converts the collection to a DataFrame.

        :return: DataFrame with job timing information
        """
        rows = []
        for job_id, info in self.items():
            rows.append({
                job_column: job_id,
                start_column: info.earliest_start,
                deadline_column: info.deadline
            })
        return pd.DataFrame(rows)

    @classmethod
    def from_dataframe(
            cls, df: pd.DataFrame, job_column: str = "Job",
            start_column: str = "Ready Time", deadline_column: str = "Deadline"):
        """
        Creates a JobInformationCollection from a DataFrame.

        Removes duplicate jobs (keeps the first occurrence).

        :param df: DataFrame with columns for job ID, earliest start, and deadline
        :return: JobInformationCollection instance
        """
        # Duplikate entfernen basierend auf der Job-ID
        df_clean = df.drop_duplicates(subset=[job_column], keep="first")

        obj = cls()
        for _, row in df_clean.iterrows():
            obj.add_job(
                job_id=str(row[job_column]),
                earliest_start=int(row[start_column]),
                deadline=int(row[deadline_column])
            )
        return obj


if __name__ == "__main__":
    # Beispiel-DataFrame mit Duplikaten
    df_jobs = pd.DataFrame([
        {"Job": "J1", "Ready Time": 0, "Deadline": 10},
        {"Job": "J2", "Ready Time": 2, "Deadline": 12},
        {"Job": "J1", "Ready Time": 1, "Deadline": 11},  # Duplikat, wird ignoriert
        {"Job": "J3", "Ready Time": 5, "Deadline": 15}
    ])

    collection = JobInformationCollection.from_dataframe(df_jobs)

    print("=== JobInformationCollection ===")
    for job_id, info in collection.items():
        print(f"Job {job_id}: Start={info.earliest_start}, Deadline={info.deadline}")