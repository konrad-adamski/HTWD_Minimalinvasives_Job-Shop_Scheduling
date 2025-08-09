from __future__ import annotations
import pandas as pd

from decimal import Decimal
from typing import List, Union
from sqlalchemy.orm import joinedload

from src.domain.orm_models import Routing, RoutingSource, Job, Machine
from src.domain.orm_setup import SessionLocal


# RoutingQuery --------------------------------------------------------------------------------------------------------
class RoutingQuery:
    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated.")

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
    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated.")

    """
    @classmethod
    def _get_by_field(cls, field_name: str, field_value: Union[str, int, Decimal]) -> List[Job]:
        if field_name not in Job.__mapper__.columns.keys():  # type: ignore[attr-defined]
            raise ValueError(f"Field '{field_name}' is not a valid column in Job.")

        with SessionLocal() as session:
            query = session.query(Job).options(
                joinedload(getattr(Job, "routing")).joinedload(getattr(Routing, "operations")),
            )
            jobs = query.filter(getattr(Job, field_name) == field_value).all()
            session.expunge_all()
            return list(jobs)

    @classmethod
    def get_by__routing_id(cls, routing_id: str) -> List[Job]:
        return cls._get_by_field("routing_id", routing_id)

    @classmethod
    def get_by_max_bottleneck_utilization(cls, max_bottleneck_utilization: Decimal) -> List[Job]:
        return cls._get_by_field("max_bottleneck_utilization", max_bottleneck_utilization)

    """

    @classmethod
    def _get_by_source_name_and_field(
            cls, source_name: str, field_name: str, field_value: Union[str, int, Decimal]) -> List[Job]:
        """
        Retrieve jobs filtered by a required RoutingSource name and an additional Job field.

        :param source_name: Name of the RoutingSource (via Job.routing.routing_source.name).
        :param field_name: Name of the Job column to filter on.
        :param field_value: Value for the Job column filter.
        :return: List of Job instances with routing and operations eagerly loaded.
        """
        if field_name not in Job.__mapper__.columns.keys():  # type: ignore[attr-defined]
            raise ValueError(f"Field '{field_name}' is not a valid column in Job.")

        with SessionLocal() as session:
            query = (
                session.query(Job)
                .join(Job.routing)  # Job -> Routing
                .join(Routing.routing_source)  # Routing -> RoutingSource
                .filter(RoutingSource.name == source_name)
                .filter(getattr(Job, field_name) == field_value)
                .options(
                    joinedload(getattr(Job, "routing"))
                    .joinedload(getattr(Routing, "operations"))
                )
                .order_by(Job.arrival)
            )
            jobs = query.all()
            session.expunge_all()
            return list(jobs)

    @classmethod
    def get_by_source_name_and_routing_id(cls, source_name: str, routing_id: str) -> List[Job]:
        return cls._get_by_source_name_and_field(source_name, "routing_id", routing_id)

    @classmethod
    def get_by_source_name_and_max_bottleneck_utilization(
            cls, source_name: str,max_bottleneck_utilization: Decimal) -> List[Job]:
        return cls._get_by_source_name_and_field(
            source_name= source_name,
            field_name="max_bottleneck_utilization",
            field_value=max_bottleneck_utilization
        )

    @staticmethod
    def update_job_due_dates_from_df(df: pd.DataFrame, job_column="Job", due_date_column="Due Date"):
        with SessionLocal() as session:
            for _, row in df.iterrows():
                job_id = row[job_column]
                new_due_date = row[due_date_column]

                job = session.get(Job, job_id)
                if job:
                    job.due_date = new_due_date

            session.commit()


# MachineQuery -------------------------------------------------------------------------------------------------------
class MachineQuery:
    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated.")

    @staticmethod
    def get_machines(source_name: str, max_bottleneck_utilization: Decimal) -> list[Machine]:
        """
        Retrieve all machines for a given routing source and max bottleneck utilization,
        with the RoutingSource eagerly loaded.

        :param source_name: Name of the routing source.
        :param max_bottleneck_utilization: Max bottleneck utilization to filter machines.
        :return: List of Machine instances.
        """
        with SessionLocal() as session:
            machines = (
                session.query(Machine)
                .join(Machine.source)
                .filter(
                    RoutingSource.name == source_name,
                    Machine.max_bottleneck_utilization == max_bottleneck_utilization
                )
                .options(joinedload(getattr(Machine, "source")))
                .order_by(Machine.name)
                .all()
            )

            session.expunge_all()
            return list(machines)


