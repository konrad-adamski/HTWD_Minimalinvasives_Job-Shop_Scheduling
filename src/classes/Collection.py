from __future__ import annotations
from collections import UserDict
from dataclasses import dataclass
from typing import Optional, List, Union

import copy
import pandas as pd

from src.classes.orm_models import JobOperation, LiveJob, Job, Routing


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

    def add_operation(
        self, job_id: str, routing_id: Optional[str],position_number: int,
        machine_name: str, duration: int, start: Optional[int] = None, end: Optional[int] = None,
        arrival: Optional[int] = None, deadline: Optional[int] = None):
        """
        Fügt eine Operation zu einem bestehenden oder neuen LiveJob hinzu.
        """

        if job_id not in self.data:
            self.data[job_id] = LiveJob(
                id=job_id,
                routing_id=routing_id,
                arrival=arrival,
                deadline=deadline,
            )
        job_op = JobOperation(
            job = self.data[job_id],
            position_number=position_number,
            machine_name=machine_name,
            start=start,
            duration=duration,
            end=end
        )
        self.data[job_id].operations.append(job_op)


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
                deadline=op.job_deadline,
            )
        else:
            job: LiveJob = self.data[job_id]

        job.add_operation_instance(op, new_start, new_duration, new_end)


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
                    result.add_operation(
                        job_id=op.job_id,
                        routing_id=op.routing_id,
                        position_number=op.position_number,
                        machine_name=op.machine_name,
                        duration=op.duration,
                        start=op.start,
                        end=op.end,
                        arrival=op.job_arrival,
                        deadline=op.job_deadline
                    )

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

    def get_latest_deadline(self) -> int:
        """
        Gibt die späteste Deadline aller Jobs zurück.
        Raises:
            ValueError: Wenn keine Deadlines gesetzt sind.
        """
        deadlines = [job.deadline for job in self.values() if job.deadline is not None]
        if not deadlines:
            raise ValueError("Keine Deadlines in der LiveJobCollection gesetzt.")
        return max(deadlines)


    # für solver info
    def count_operations(self) -> int:
        return sum(len(job.operations) for job in self.values())


    # DataFrame ----------------------------------------------------------------

    def to_operations_dataframe(
            self, job_column: str = "Job", routing_column: str = "Routing_ID", position_column: str = "Operation",
            machine_column: str = "Machine", start_column: str = "Start", duration_column: str = "Processing Time",
            end_column: str = "End", earliest_start_column: str = "Ready Time",
            deadline_column: str = "Deadline") -> pd.DataFrame:
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
                    earliest_start_column: job.earliest_start,
                    deadline_column: job.deadline
                })
        return pd.DataFrame(records)

    def to_jobs_dataframe(
            self, job_column: str = "Job", routing_column: str = "Routing_ID",
            arrival_column: str = "Arrival", earliest_start_column: str = "Ready Time",
            deadline_column: str = "Deadline") -> pd.DataFrame:
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
                deadline_column: job.deadline
            })
        return pd.DataFrame(records)


    def to_last_ops_dataframe(
            self, job_column: str = "Job", routing_column: str = "Routing_ID", position_column: str = "Operation",
            machine_column: str = "Machine", start_column: str = "Start", duration_column: str = "Processing Time",
            end_column: str = "End", earliest_start_column: str = "Ready Time",
            deadline_column: str = "Deadline") -> pd.DataFrame:

        job_sum_durations = {job.id: job.sum_duration for job in self.values()}
        last_job_ops_collection = self._get_last_operations_collection()

        df = last_job_ops_collection.to_operations_dataframe(
            job_column=job_column, routing_column=routing_column, position_column=position_column,
            machine_column=machine_column, start_column=start_column, duration_column=duration_column,
            end_column=end_column, earliest_start_column=earliest_start_column, deadline_column=deadline_column
        )
        df[f"Total {duration_column}"] = df[job_column].map(job_sum_durations)
        return df


    # for Simulation (Tests)
    @classmethod
    def from_operations_dataframe(
            cls, df: pd.DataFrame, job_column: str = "Job", routing_column: str = "Routing_ID",
            position_column: str = "Operation", machine_column: str = "Machine",
            start_column: str = "Start", duration_column: str = "Processing Time",
            end_column: str = "End", arrival_column:str ="Arrival",
            deadline_column:str  = "Deadline") -> LiveJobCollection:
        """
        Erstellt eine LiveJobCollection aus einem DataFrame mit Zeilen für einzelne Operationen.
        """
        obj = cls()

        has_routing_column = routing_column in df.columns
        has_arrival_column = arrival_column in df.columns
        has_deadline_column = deadline_column in df.columns
        for _, row in df.iterrows():
            routing_id = str(row[routing_column]) if has_routing_column and pd.notna(row[routing_column]) else None
            arrival = int(row[arrival_column]) if has_arrival_column and pd.notna(row[arrival_column]) else None
            deadline = int(row[deadline_column]) if has_deadline_column and pd.notna(row[deadline_column]) else None

            obj.add_operation(
                job_id=str(row[job_column]),
                routing_id=routing_id,
                position_number=int(row[position_column]),
                machine_name=str(row[machine_column]),
                duration=int(row[duration_column]),
                start=int(row[start_column]) if pd.notna(row[start_column]) else None,
                end=int(row[end_column]) if pd.notna(row[end_column]) else None,
                arrival=arrival,
                deadline=deadline
            )

        obj.sort_operations()
        return obj

