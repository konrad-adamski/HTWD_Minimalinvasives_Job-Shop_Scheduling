from __future__ import annotations
from collections import UserDict
from typing import List, Optional, Union
import pandas as pd
from sqlalchemy import or_, and_
from sqlalchemy.orm import joinedload, InstrumentedAttribute

from src.classes.orm_models import Routing, RoutingSource, RoutingOperation, Job
from src.classes.orm_setup import SessionLocal

# RoutingCollection ---------------------------------------------------------------------------------------------------
class RoutingsCollection(UserDict):
    def __init__(self, routings: Optional[List[Routing]] = None):
        """Initialize collection from an optional list of Routing objects (keyed by ID)."""
        routings = routings or []
        super().__init__({routing.id: routing for routing in routings})

    def get_routing(self, routing_id: str) -> Optional[Routing]:
        """Find a Routing by its ID."""
        return self.data.get(routing_id)

    def get_routings(self) -> List[Routing]:
        return list(self.values())

    def sort_operations(self):
        """Sort operations in each routing by position number."""
        for routing in self.values():
            routing.operations.sort(key=lambda op: op.position_number)

    @classmethod
    def from_dataframe(cls, df_routings: pd.DataFrame,
                       routing_column: str = "Routing_ID",
                       operation_column: str = "Operation",
                       machine_column: str = "Machine",
                       duration_column: str = "Processing Time",
                       source: Optional[RoutingSource] = None) -> RoutingsCollection:
        """
        Create a RoutingCollection from a DataFrame containing one or more routings.
        """
        routings = []

        for routing_id, group in df_routings.groupby(routing_column):
            df_clean = group.drop_duplicates(subset=[routing_column, operation_column], keep="first")
            routing_id_str = str(routing_id)
            new_routing = Routing(id=routing_id_str, routing_source=source, operations=[])

            for _, row in df_clean.iterrows():
                step_nr = int(row[operation_column])
                machine = str(row[machine_column])
                duration = int(row[duration_column])

                new_routing.operations.append(
                    RoutingOperation(
                        routing_id=routing_id_str,
                        position_number=step_nr,
                        machine=machine,
                        duration=duration
                    )
                )

            new_routing.operations.sort(key=lambda op: op.position_number)
            routings.append(new_routing)

        return cls(routings)

    @classmethod
    def from_db_by_source_name(cls, source_name: str) -> RoutingsCollection:
        """
        Retrieve all routing entries with the given routing source name.

        :param source_name: Name of the routing source to filter by.
        :return: List of Routing instances with their source and operations loaded.
        """
        with SessionLocal() as session:
            routings = (
                session.query(Routing)
                .join(Routing.routing_source)
                .filter(RoutingSource.name == source_name)
                .options(
                    joinedload(getattr(Routing, "routing_source")),
                    joinedload(getattr(Routing, "operations"))
                )
                .all()
            )
            session.expunge_all()
            return cls(routings)


    def to_dataframe(self) -> pd.DataFrame:
        """
        Export all routing operations to a flat DataFrame.

        Columns: Routing_ID, Operation, Machine, Processing Time
        """
        records = []

        for routing in self.values():
            for op in routing.operations:
                records.append({
                    "Routing_ID": routing.id,
                    "Operation": op.position_number,
                    "Machine": op.machine,
                    "Processing Time": op.duration
                })

        return pd.DataFrame(records)


# JobsCollection -------------------------------------------------------------------------------------------------------
class JobsCollection(UserDict):
    def __init__(self, jobs: List[Job]):
        """Initialize collection from a list of Job objects using their ID as key."""
        super().__init__({job.id: job for job in jobs})

    def get_job(self, job_id: str) -> Optional[Job]:
        """Return job by ID, or None if not found."""
        return self.data.get(job_id)

    def all_jobs(self) -> List[Job]:
        """Return all jobs as a list."""
        return list(self.values())

    def to_dataframe(
            self, job_column: str = "Job", routing_column: str = "Routing_ID",
            arrival_column = "Arrival", earliest_start_column = "Ready Time",
            deadline_column = "Deadline") -> pd.DataFrame:
        """Export all jobs to a DataFrame."""
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
    def _from_db_by_field(cls, field_name: str, field_value: Union[str, int]) -> JobsCollection:
        if field_name not in Job.__mapper__.columns.keys():  # type: ignore[attr-defined]
            raise ValueError(f"Field '{field_name}' is not a valid column in Job.")

        with SessionLocal() as session:
            query = session.query(Job).options(
                joinedload(getattr(Job, "routing")).joinedload(getattr(Routing, "operations")),
                joinedload(getattr(Job, "experiment")) #,
          #      joinedload(getattr(Job, "schedule_operations")),
          #      joinedload(getattr(Job, "simulation_operations"))

            )
            jobs = query.filter(getattr(Job, field_name) == field_value).all()
            session.expunge_all()
            return cls(jobs)

    @classmethod
    def from_db_by_routing_id(cls, routing_id: str) -> JobsCollection:
        return cls._from_db_by_field("routing_id", routing_id)

    @classmethod
    def from_db_by_experiment_id(cls, experiment_id: int) -> JobsCollection:
        return cls._from_db_by_field("experiment_id", experiment_id)


    @classmethod
    def from_db_by_earliest_start_or_ids(
            cls, experiment_id: int, earliest_start: int,
            job_ids: Optional[List[str]] = None) -> JobsCollection:
        """
        Retrieve all jobs for a given experiment where either the earliest start time matches
        the specified value or the job ID is in the given list.

        :param experiment_id: ID of the experiment to filter jobs by.
        :param earliest_start: Earliest start time to match.
        :param job_ids: Optional list of job IDs to include in the result.
        :return: List of Job instances with routing and operation details loaded.
        """

        with SessionLocal() as session:
            conditions = [Job.experiment_id == experiment_id]

            if job_ids:
                job_id_attr: InstrumentedAttribute = getattr(Job, "id")
                conditions.append(
                    or_(Job.earliest_start == earliest_start, job_id_attr.in_(job_ids))
                )
            else:
                conditions.append(Job.earliest_start == earliest_start)

            query = session.query(Job).filter(and_(*conditions)).options(
                joinedload(getattr(Job, "routing")).joinedload(getattr(Routing, "operations")),
                joinedload(getattr(Job, "experiment"))
            )

            jobs = query.all()
            session.expunge_all()
            return cls(jobs)

