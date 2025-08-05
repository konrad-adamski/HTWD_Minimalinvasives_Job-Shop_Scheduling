from __future__ import annotations
from collections import UserDict
from typing import List, Optional, Union
import pandas as pd
from sqlalchemy import or_, and_, update
from sqlalchemy.orm import joinedload, InstrumentedAttribute

from src.classes.orm_models import Routing, RoutingSource, RoutingOperation, Job
from src.classes.orm_setup import SessionLocal

# RoutingQuery --------------------------------------------------------------------------------------------------------
class RoutingQuery:

    @staticmethod
    def insert_from_dataframe(df_routings: pd.DataFrame,
                       routing_column: str = "Routing_ID",
                       operation_column: str = "Operation",
                       machine_column: str = "Machine",
                       duration_column: str = "Processing Time",
                       source: Optional[RoutingSource] = None):
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
                machine_name = str(row[machine_column])
                duration = int(row[duration_column])

                new_routing.operations.append(
                    RoutingOperation(
                        routing_id=routing_id_str,
                        position_number=step_nr,
                        machine_name=machine_name,
                        duration=duration
                    )
                )

            new_routing.operations.sort(key=lambda op: op.position_number)
            routings.append(new_routing)

        with SessionLocal() as session:
            session.add_all(routings)
            session.commit()


    @staticmethod
    def get_by_source_name(source_name: str) -> List[Routing]:
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
            return list(routings)



# JobQuery -----------------------------------------------------------------------------------------------------------
class JobQuery:

    @classmethod
    def _get_by_field(cls, field_name: str, field_value: Union[str, int]) -> List[Job]:
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
            return list(jobs)

    @classmethod
    def get_by__routing_id(cls, routing_id: str) -> List[Job]:
        return cls._get_by_field("routing_id", routing_id)

    @classmethod
    def get_by_experiment_id(cls, experiment_id: int) -> List[Job]:
        return cls._get_by_field("experiment_id", experiment_id)


    @classmethod
    def get_by_earliest_start_or_ids(
            cls, experiment_id: int, earliest_start: int,
            job_ids: Optional[List[str]] = None) -> List[Job]:
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
            return list(jobs)

    @staticmethod
    def update_job_deadlines_from_df(df: pd.DataFrame, job_column="Job", deadline_column="Deadline"):
        with SessionLocal() as session:
            for _, row in df.iterrows():
                job_id = row[job_column]
                new_deadline = row[deadline_column]

                job = session.get(Job, job_id)
                if job:
                    job.deadline = new_deadline

            session.commit()

