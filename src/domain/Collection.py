from __future__ import annotations
from collections import UserDict
from dataclasses import dataclass
from typing import Optional, List, Union

import pandas as pd

from src.domain.orm_models import JobOperation, LiveJob, Job, Routing


class RoutingCollection(UserDict[str, Routing]):
    def __init__(self, initial: List[Routing]):
        super().__init__()

        if initial:
            for routing in initial:
                self.data[routing.id] = routing

    def to_dataframe(self) -> pd.DataFrame:
        records = []

        for routing in self.data.values():
            for op in routing.operations:
                records.append({
                    "Routing_ID": routing.id,
                    "Operation": op.position_number,
                    "Machine": op.machine_name,
                    "Processing Time": op.duration
                })

        return pd.DataFrame(records)


@dataclass
class LiveJobCollection(UserDict[str, LiveJob]):
    """
    Verwaltet eine Sammlung von LiveJobs, adressiert über job_id.
    """
    def __init__(self, initial: Optional[Union[List[LiveJob], List[Job]]] = None):
        super().__init__()

        if initial:
            for job in initial:
                job_template = LiveJob.copy_from(job)
                self.data[job_template.id] = job_template


    def add_operation_instance(self, op: JobOperation, new_start: Optional[float] = None,
            new_duration: Optional[float] = None, new_end: Optional[float] = None):
        """
        Adds a complete JobWorkflowOperation instance to the collection.
        """
        job_id = op.job.id

        if job_id not in self.data:
            job = LiveJob(
                id=job_id,
                routing_id=op.routing_id,
                arrival=op.job_arrival,
                due_date=op.job_due_date
            )
        else:
            job: LiveJob = self.data[job_id]

        job.add_operation_instance(op, new_start, new_duration, new_end)
        self.data[job_id] = job

    def sort_operations(self):
        for job in self.values():
            job.operations.sort(key=lambda op: op.position_number)

    def sort_jobs_by_arrival(self) -> None:
        """
        Sortiert die LiveJobCollection intern nach arrival-Zeitpunkt der Jobs.
        Jobs ohne arrival stehen am Ende.
        """
        jobs_list = list(self.values())
        jobs_list.sort(key=lambda job: (job.arrival is None, job.arrival))
        sorted_data = {job.id: job for job in jobs_list}
        self.data = sorted_data

    def sort_jobs_by_id(self) -> None:
        """
        Sortiert die LiveJobCollection intern ausschließlich nach dem letzten Block der Job-ID.
        Beispiel: '01-07500-0001' < '01-07500-0010' < '01-07500-1000'
        """
        jobs_list = list(self.values())
        jobs_list.sort(key=lambda job: int(job.id.split("-")[-1]))
        self.data = {job.id: job for job in jobs_list}


    def get_all_jobs(self) -> List[LiveJob]:
        return list(self.values())

    def get_unique_machine_names(self) -> set:
        """
        Gibt die Menge aller in den Operationen verwendeten Maschinen zurück.
        """
        machines = {
            op.machine_name
            for job in self.values()
            for op in job.operations
            if op.machine_name is not None
        }
        return machines

    def get_subset_by_earliest_start(self, earliest_start: int) -> LiveJobCollection:
        """
        Returns all jobs whose earliest_start matches the given value.

        :param earliest_start: Time threshold for selecting full jobs
        :return: Filtered LiveJobCollection with complete jobs
        """
        subset = LiveJobCollection()
        for job in self.values():
            if job.earliest_start == earliest_start:
                subset[job.id] = job
        return subset


    def _get_last_operations_collection(self) -> LiveJobCollection:
        """
        Gibt eine neue LiveJobCollection mit nur der letzten Operation pro Job zurück.
        """
        result = LiveJobCollection()

        for job in self.values():
            if not job.operations:
                continue
            last_op = max(job.operations, key=lambda op: op.position_number)
            result.add_operation_instance(last_op)
        return result


    # Operatorüberladungen --------------------------------------------------------
    @classmethod
    def _subtract_by_job_operation_collection(
            cls, main: LiveJobCollection, exclude: LiveJobCollection) -> LiveJobCollection:
        """
        Gibt eine neue LiveJobCollection zurück, die nur jene Operationen aus 'main' enthält,
        deren (job_id, position_number) nicht in der 'exclude'-Collection vorkommen.

        Für jeden verbleibenden Job wird bei Bedarf ein neuer LiveJob mit zugehörigen
        Operationen erzeugt.
        """
        result = cls()

        # 1) Setze die auszuschließenden (job_id, position_number)-Paare
        excluded_pairs = {
            (op.job_id, op.position_number)
            for job in exclude.values()
            for op in job.operations
        }

        # 2) Iteriere über alle Operationen in main
        for job in main.values():
            for op in job.operations:
                if (op.job_id, op.position_number) not in excluded_pairs:
                    result.add_operation_instance(op)

        result.sort_operations()
        return result


    def __truediv__(self, other: LiveJobCollection) -> LiveJobCollection:
        return self.__class__._subtract_by_job_operation_collection(main=self, exclude=other)


    @classmethod
    def _merge_collections(cls, main: LiveJobCollection, include: LiveJobCollection) -> LiveJobCollection:
        """
        Merges two LiveJobCollections into a new one.
        If a job or operation exists in both, the version from 'main' takes precedence.

        :param main: Primary LiveJobCollection (priority)
        :param include: Secondary LiveJobCollection (merged if not in main)
        :return: A new merged LiveJobCollection
        """
        result = cls()

        # Zuerst alles aus Main übernehmen
        for job_id, job_main in main.items():
            result[job_id] = LiveJob.copy_from(job_main)

        # Dann fehlende Jobs + fehlende Operationen aus Include ergänzen
        for job_id, job_include in include.items():
            if job_id not in result:
                result[job_id] = LiveJob.copy_from(job_include)
            else:
                # Nur neue Operationen übernehmen (nach position_number)
                existing_ops = {op.position_number for op in result[job_id].operations}
                for op in job_include.operations:
                    if op.position_number not in existing_ops:
                        result[job_id].add_operation_instance(op)
        result.sort_operations()
        result.sort_jobs_by_arrival()
        return result

    def __add__(self, other: LiveJobCollection) -> LiveJobCollection:
        """
        Combines two LiveJobCollections using the + operator.
        If jobs or operations exist in both, the version from 'self' takes precedence.

        :param other: Another LiveJobCollection
        :return: A new merged LiveJobCollection
        """
        return self.__class__._merge_collections(self, other)


    # für solver-model --------------------------------------------------------
    def get_total_duration(self) -> int:
        """
        Gibt die Gesamtdauer aller Jobs zurück (Summe aller job.sum_duration).
        """
        return sum(job.sum_duration for job in self.values())

    def get_latest_due_date(self) -> Optional[int]:
        due_dates = [job.due_date for job in self.values() if job.due_date is not None]
        if not due_dates:
            return None
        return max(due_dates)

    def get_latest_arrival(self) -> Optional[int]:

        arrivals = [job.arrival for job in self.values() if job.arrival is not None]
        if not arrivals:
            return None
        return max(arrivals)

    def get_latest_earliest_start(self) -> Optional[int]:

        earliest_starts = [job.earliest_start for job in self.values() if job.earliest_start is not None]
        if not earliest_starts:
            return None
        return max(earliest_starts)


    # für solver info
    def count_operations(self) -> int:
        return sum(len(job.operations) for job in self.values())


    # für LP ---------------------
    def get_all_operations_on_machine(self, machine_name: str) -> List[JobOperation]:
        """
        Gibt alle Operationen zurück, die auf der angegebenen Maschine stattfinden.

        :param machine_name: Name der Maschine
        :return: Liste der passenden JobOperation-Objekte
        """
        return [
            op
            for job in self.values()
            for op in job.operations
            if op.machine_name == machine_name
        ]


    # DataFrame ----------------------------------------------------------------
    def to_operations_dataframe(
            self, job_column: str = "Job", routing_column: str = "Routing_ID", position_column: str = "Operation",
            machine_column: str = "Machine", start_column: str = "Start", duration_column: str = "Processing Time",
            end_column: str = "End", arrival_column = "Arrival", earliest_start_column: str = "Ready Time",
            due_date_column: str = "Due Date") -> pd.DataFrame:
        """
        Gibt einen DataFrame mit allen Operationen in der Collection zurück.
        Nur Jobs mit Attribut 'operations' (d.h. LiveJobs) werden berücksichtigt.
        """

        records = []

        for job in self.values():
            for op in job.operations:
                records.append({
                    job_column: job.id,
                    routing_column: job.routing_id,
                    position_column: op.position_number,
                    machine_column: op.machine_name,
                    start_column: op.start,
                    duration_column: op.duration,
                    end_column: op.end,
                    arrival_column: job.arrival,
                    earliest_start_column: job.earliest_start,
                    due_date_column: job.due_date
                })
        return pd.DataFrame(records)

    def to_waiting_time_dataframe(
            self,
            job_column: str = "Job",
            position_column: str = "Operation",
            machine_column: str = "Machine",
            request_time_column: str = "Request Time",
            granted_time_column: str = "Granted Time",
            waiting_time_column: str = "Waiting Time"
    ) -> pd.DataFrame:
        """
        Gibt einen DataFrame mit Request-/Granted-/Waiting-Zeiten je Operation zurück.
        Nutzt direkt die Felder aus JobOperation.
        """
        records = []

        for job in self.values():
            for op in job.operations:
                records.append({
                    job_column: job.id,
                    position_column: op.position_number,
                    machine_column: op.machine_name,
                    request_time_column: op.request_time_on_machine,
                    granted_time_column: op.granted_time_on_machine,
                    waiting_time_column: op.waiting_time_on_machine
                })

        return pd.DataFrame(records)


    def to_transition_time_dataframe(
            self, job_column: str = "Job", position_column: str = "Operation", machine_column: str = "Machine",
            granted_time_column: str = "Granted Time", end_column: str = "End",
            prev_end_column: str = "Prev End", transition_time_column: str = "Transition Time"):
        """
        Erstellt einen DataFrame mit Transition Times je Maschine (Backward-Variante).

        Transition Time = Granted Time der aktuellen OP minus Endzeit der nachfolgenden OP
        (in der Rückwärtsperspektive der "vorherige" Auftrag).
        """

        records = []
        for job in self.values():
            for op in job.operations:
                records.append({
                    job_column: job.id,
                    position_column: op.position_number,
                    machine_column: op.machine_name,
                    granted_time_column: op.granted_time_on_machine,
                    end_column: op.end,
                })
        df = pd.DataFrame(records)

        if df.empty:
            return None

        # Absteigend sortieren: nächster Auftrag in der Zukunft kommt zuerst
        df = df.sort_values([machine_column, granted_time_column], ascending=[True, False], kind="stable")

        # Ende der zeitlich vorherigen Operation auf derselben Maschine
        df[prev_end_column] = df.groupby(machine_column)[end_column].shift(-1)

        # Transition Time = granted_i - end_{i+1}
        df[transition_time_column] = df[granted_time_column] - df[prev_end_column]

        return df

    def to_jobs_dataframe(
            self, job_column: str = "Job", routing_column: str = "Routing_ID",
            arrival_column: str = "Arrival", earliest_start_column: str = "Ready Time",
            due_date_column: str = "Due Date") -> pd.DataFrame:
        """
        Gibt einen DataFrame mit allen Jobinformationen in der Collection zurück.
        """
        records = []
        for job in self.values():
            records.append({
                job_column: job.id,
                routing_column: job.routing_id,
                arrival_column: job.arrival,
                earliest_start_column: job.earliest_start,
                due_date_column: job.due_date
            })
        return pd.DataFrame(records)


    def to_jobs_metrics_dataframe(
            self, job_column: str = "Job", routing_column: str = "Routing_ID",
            last_position_column: str = "Last Operation", total_duration_column: str = "Total Processing Time",
            end_column: str = "End", arrival_column = "Arrival", earliest_start_column: str = "Ready Time",
            due_date_column: str = "Due Date", lateness_column: str = "Lateness", tardiness_column: str = "Tardiness",
            earliness_column: str = "Earliness", flowtime_column: Optional[str] = "Flowtime") -> pd.DataFrame:

        # Total duration (based on every operation of each job)
        job_sum_durations = {job.id: job.sum_duration for job in self.values()}

        # Collection with only the last operation of each job
        last_job_ops_collection = self._get_last_operations_collection()

        df = last_job_ops_collection.to_operations_dataframe(
            job_column=job_column, routing_column=routing_column,
            position_column=last_position_column,
            end_column=end_column, arrival_column= arrival_column,
            earliest_start_column=earliest_start_column, due_date_column=due_date_column
        )

        df[total_duration_column] = df[job_column].map(job_sum_durations)
        if flowtime_column:
            df[flowtime_column] = df[end_column] - df[earliest_start_column]

        df[lateness_column] = df[end_column] - df[due_date_column]
        df[tardiness_column] = df[lateness_column].clip(lower=0)
        df[earliness_column] = (-df[lateness_column]).clip(lower=0)
        df = df.drop(["Machine", "Start", "Processing Time"], axis=1) # default in 'to_operations_dataframe()'
        return df


    # for Simulation (Tests)
    @classmethod
    def from_operations_dataframe(
            cls, df: pd.DataFrame, job_column: str = "Job", routing_column: str = "Routing_ID",
            position_column: str = "Operation", machine_column: str = "Machine",
            start_column: str = "Start", duration_column: str = "Processing Time",
            end_column: str = "End", arrival_column:str ="Arrival",
            due_date_column:str  = "Due Date") -> LiveJobCollection:
        """
        Erstellt eine LiveJobCollection aus einem DataFrame mit Zeilen für einzelne Operationen.
        """
        obj = cls()

        has_routing_column = routing_column in df.columns
        has_arrival_column = arrival_column in df.columns
        has_due_date_column = due_date_column in df.columns

        for job_id, group in df.groupby(job_column, sort=True):

            routing_not_na = pd.notna(group[routing_column].iloc[0]) if has_routing_column else False
            routing_id = str(group[routing_column].iloc[0]) if routing_not_na else None

            arrival_not_na = pd.notna(group[arrival_column].iloc[0]) if has_arrival_column else False
            arrival = int(group[arrival_column].iloc[0]) if arrival_not_na else None

            due_date_not_na = pd.notna(group[due_date_column].iloc[0]) if has_due_date_column else False
            due_date = int(group[due_date_column].iloc[0]) if due_date_not_na else None

            job = LiveJob(
                id=str(job_id),
                routing_id=routing_id,
                arrival=arrival,
                due_date=due_date
            )

            for _, row in group.iterrows():
                operation = JobOperation(
                    job=job,
                    position_number=row[position_column],
                    machine_name=str(row[machine_column]),
                    duration=int(row[duration_column]),
                    start=int(row[start_column]) if pd.notna(row[start_column]) else None,
                    end=int(row[end_column]) if pd.notna(row[end_column]) else None,
                )
                obj.add_operation_instance(operation)

        obj.sort_operations()
        return obj

