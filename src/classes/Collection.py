from __future__ import annotations
from collections import UserDict
from dataclasses import dataclass
from typing import Optional, List, Union

import copy
import pandas as pd

from src.classes.orm_models import JobOperation, JobTemplate, Job, Routing


class RoutingCollection(UserDict):
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
                    "Machine": op.machine,
                    "Processing Time": op.duration
                })

        return pd.DataFrame(records)


@dataclass
class JobMixCollection(UserDict):
    """
    Verwaltet eine Sammlung von JobTemplates, adressiert über job_id.
    """

    def __init__(self, initial: Optional[Union[List[JobTemplate], List[Job]]] = None):
        super().__init__()

        if initial:
            for job in initial:
                self.data[job.id] = job

    def add_operation(
        self, job_id: str, routing_id: Optional[str], experiment_id: Optional[int],position_number: int,
        machine: str, duration: int, start: Optional[int] = None, end: Optional[int] = None,
        arrival: Optional[int] = None, deadline: Optional[int] = None):
        """
        Fügt eine Operation zu einem bestehenden oder neuen JobTemplate hinzu.
        """

        if job_id not in self.data:
            self.data[job_id] = JobTemplate(
                id=job_id,
                routing_id=routing_id,
                experiment_id=experiment_id,
                arrival=arrival,
                deadline=deadline,
            )
        job_op = JobOperation(
            job = self.data[job_id],
            position_number=position_number,
            machine=machine,
            start=start,
            duration=duration,
            end=end
        )
        self.data[job_id].operations.append(job_op)

    def add_operation_instance(self, op: JobOperation):
        """
        Adds a complete JobWorkflowOperation instance to the collection.
        """
        job_id = op.job.id
        if job_id not in self.data:
            self.data[job_id] = JobTemplate(
                id=job_id,
                routing_id=op.routing_id,
                experiment_id=op.experiment_id,
                arrival=op.job_arrival,
                deadline=op.job_deadline,
            )

        self.data[job_id].operations.append(op)

    def sort_operations(self):
        for job in self.values():
            job.operations.sort(key=lambda op: op.position_number)

    def sort_jobs_by_arrival(self) -> None:
        """
        Sortiert die JobMixCollection intern nach arrival-Zeitpunkt der Jobs.
        Jobs ohne arrival stehen am Ende.
        """
        jobs_list = list(self.values())
        jobs_list.sort(key=lambda job: (job.arrival is None, job.arrival))
        sorted_data = {job.id: job for job in jobs_list}
        self.data = sorted_data



    def get_all_jobs(self) -> [Union[List[JobTemplate], List[Job]]]:
        return list(self.values())

    def get_unique_machines(self) -> set:
        """
        Gibt die Menge aller in den Operationen verwendeten Maschinen zurück.
        """
        machines = {
            op.machine
            for job in self.values()
            for op in job.operations
            if op.machine is not None
        }
        return machines


    @classmethod
    def from_schedule_dataframe(
            cls, df: pd.DataFrame, job_column: str = "Job", routing_column: str = "Routing_ID",
            position_column: str = "Operation", machine_column: str = "Machine",
            start_column: str = "Start", duration_column: str = "Processing Time",
            end_column: str = "End") -> JobMixCollection:
        """
        Erstellt eine JobMixCollection aus einem DataFrame mit Zeilen für einzelne Operationen.
        """
        obj = cls()

        has_routing_column = routing_column in df.columns
        for _, row in df.iterrows():
            job_id = str(row[job_column])
            routing_id = str(row[routing_column]) if has_routing_column and pd.notna(row[routing_column]) else None
            position_number = int(row[position_column])
            machine = str(row[machine_column])
            duration = int(row[duration_column])
            start = int(row[start_column]) if pd.notna(row[start_column]) else None
            end = int(row[end_column]) if pd.notna(row[end_column]) else None

            obj.add_operation(
                job_id=job_id,
                routing_id=routing_id,
                experiment_id=None,
                position_number=position_number,
                machine=machine,
                duration=duration,
                start=start,
                end=end,
                arrival=None,
                deadline=None
            )

        obj.sort_operations()
        return obj

    def to_dataframe(
            self, job_column: str = "Job", routing_column: str = "Routing_ID", position_column: str = "Operation",
            machine_column: str = "Machine", start_column: str = "Start", duration_column: str = "Processing Time",
            end_column: str = "End", earliest_start_column: str = "Ready Time",
            deadline_column: str = "Deadline") -> pd.DataFrame:
        """
        Gibt einen DataFrame mit allen Operationen in der Collection zurück.
        Nur Jobs mit Attribut 'operations' (d.h. JobTemplates) werden berücksichtigt.
        """

        records = []

        for job in self.values():
            for op in job.operations:
                records.append({
                    job_column: job.id,
                    routing_column: job.routing_id,
                    position_column: op.position_number,
                    machine_column: op.machine,
                    start_column: op.start,
                    duration_column: op.duration,
                    end_column: op.end,
                    earliest_start_column: job.earliest_start,
                    deadline_column: job.deadline
                })
        return pd.DataFrame(records)

    def to_information_dataframe(
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


    @classmethod
    def _subtract_by_job_operation_collection(
            cls, main: JobMixCollection, exclude: JobMixCollection) -> JobMixCollection:
        """
        Gibt eine neue JobMixCollection zurück, die nur jene Operationen aus 'main' enthält,
        deren (job_id, position_number) nicht in der 'exclude'-Collection vorkommen.

        Für jeden verbleibenden Job wird bei Bedarf ein neuer JobTemplate mit zugehörigen
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
                        experiment_id=op.experiment_id,
                        position_number=op.position_number,
                        machine=op.machine,
                        duration=op.duration,
                        start=op.start,
                        end=op.end,
                        arrival=op.job_arrival,
                        deadline=op.job_deadline
                    )

        result.sort_operations()
        return result

    def __truediv__(self, other: JobMixCollection) -> JobMixCollection:
        return self.__class__._subtract_by_job_operation_collection(main=self, exclude=other)

    def get_last_operations_only(self) -> JobMixCollection:
        """
        Gibt eine neue JobMixCollection mit nur der letzten Operation pro Job zurück.
        """
        result = JobMixCollection()

        for job in self.values():
            if not job.operations:
                continue
            last_op = max(job.operations, key=lambda op: op.position_number)
            result.add_operation(
                job_id=last_op.job_id,
                routing_id=last_op.routing_id,
                experiment_id=last_op.experiment_id,
                position_number=last_op.position_number,
                machine=last_op.machine,
                duration=last_op.duration,
                start=last_op.start,
                end=last_op.end,
                arrival=last_op.job_arrival,
                deadline=last_op.job_deadline
            )

        result.sort_operations()
        return result


    def to_last_ops_dataframe(
                self, job_column: str = "Job", routing_column: str = "Routing_ID", position_column: str = "Operation",
                machine_column: str = "Machine", start_column: str = "Start", duration_column: str = "Processing Time",
                end_column: str = "End", earliest_start_column: str = "Ready Time",
                deadline_column: str = "Deadline") -> pd.DataFrame:

            job_sum_durations = {job.id: job.sum_duration for job in self.values()}
            last_job_ops_collection = self.get_last_operations_only()

            records = []

            for job in last_job_ops_collection.values():
                for op in job.operations:
                    records.append({
                        job_column: job.id,
                        routing_column: job.routing_id,
                        position_column: op.position_number,
                        machine_column: op.machine,
                        start_column: op.start,
                        duration_column: op.duration,
                        end_column: op.end,
                        earliest_start_column: job.earliest_start,
                        deadline_column: job.deadline,
                        f"Total {duration_column}": job_sum_durations[job.id],
                    })
            df = pd.DataFrame(records).sort_values(by=[start_column])
            return df

    def get_subset_by_earliest_start(self, earliest_start: int) -> JobMixCollection:
        """
        Returns all jobs whose earliest_start matches the given value.

        :param earliest_start: Time threshold for selecting full jobs
        :return: Filtered JobMixCollection with complete jobs
        """
        subset = JobMixCollection()
        for job in self.values():
            if job.earliest_start == earliest_start:
                subset[job.id] = job
        return subset

    @classmethod
    def merge_collections(cls, a: JobMixCollection, b: JobMixCollection) -> JobMixCollection:
        """
        Merges two JobMixCollections into a new one.
        If a job or operation exists in both, the version from 'a' takes precedence.

        :param a: Primary JobMixCollection (priority)
        :param b: Secondary JobMixCollection (merged if not in a)
        :return: A new merged JobMixCollection
        """
        result = cls()

        # Zuerst alles aus A übernehmen
        for job_id, job in a.items():
            result[job_id] = JobTemplate(
                id=job.id,
                routing_id=job.routing_id,
                experiment_id=job.experiment_id,
                arrival=job.arrival,
                deadline=job.deadline,
                operations=[copy.deepcopy(op) for op in job.operations]  # Kopieren für Sicherheit
            )

        # Dann fehlende Jobs + fehlende Operationen aus B ergänzen
        for job_id, job_b in b.items():
            if job_id not in result:
                result[job_id] = JobTemplate(
                    id=job_b.id,
                    routing_id=job_b.routing_id,
                    experiment_id=job_b.experiment_id,
                    arrival=job_b.arrival,
                    deadline=job_b.deadline,
                    operations=[copy.deepcopy(op) for op in job_b.operations]
                )
            else:
                # Nur neue Operationen übernehmen (nach position_number)
                existing_ops = {op.position_number for op in result[job_id].operations}
                for op in job_b.operations:
                    if op.position_number not in existing_ops:
                        result[job_id].operations.append(copy.deepcopy(op))

        result.sort_operations()
        return result

    def __add__(self, other: JobMixCollection) -> JobMixCollection:
        """
        Combines two JobMixCollections using the + operator.
        If jobs or operations exist in both, the version from 'self' takes precedence.

        :param other: Another JobMixCollection
        :return: A new merged JobMixCollection
        """
        return self.__class__.merge_collections(self, other)


    # für solver-model -----------------------------------------------------------------------------------------------
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
            raise ValueError("Keine Deadlines in der JobMixCollection gesetzt.")
        return max(deadlines)
