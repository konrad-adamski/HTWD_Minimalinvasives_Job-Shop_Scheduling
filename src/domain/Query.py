from __future__ import annotations
import pandas as pd

from decimal import Decimal
from typing import List, Union, Iterable, Tuple
from sqlalchemy.orm import joinedload

from src.domain.orm_models import Routing, RoutingSource, Job, Machine, Experiment, ScheduleOperation, ScheduleJob, \
    LiveJob, SimulationJob, SimulationOperation
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

    @classmethod
    def _get_by_source_name_and_job_filters(
            cls,
            source_name: str,
            job_filters: dict[str, Union[str, int, Decimal]]
    ) -> List[Job]:
        """
        Retrieve jobs filtered by a required RoutingSource name and one or more Job fields.
        Supports operators via suffix:
            __eq   -> ==
            __lte  -> <=
            __lt   -> <
            __gte  -> >=
            __gt   -> >
            __ne   -> !=
        Without suffix -> ==

        :param source_name: Name of the RoutingSource (via Job.routing.routing_source.name).
        :param job_filters: Dict of {field_name or field_name__op: field_value}
                            to filter Job columns.
        :return: List of Job instances with routing and operations eagerly loaded.
        """
        # 1) Gültige Spaltennamen prüfen (ohne Operator-Suffix)
        valid_columns = Job.__mapper__.columns.keys()  # type: ignore[attr-defined]
        for raw_field in job_filters.keys():
            base_field = raw_field.split("__", 1)[0]
            if base_field not in valid_columns:
                raise ValueError(f"Field '{base_field}' is not a valid column in Job.")

        # 2) Abfrage aufbauen
        with SessionLocal() as session:
            query = (
                session.query(Job)
                .join(Job.routing)
                .join(Routing.routing_source)
                .filter(RoutingSource.name == source_name)
            )

            # 3) Dynamische Filter hinzufügen
            for raw_field, value in job_filters.items():
                if "__" in raw_field:
                    field_name, op_suffix = raw_field.split("__", 1)
                else:
                    field_name, op_suffix = raw_field, None

                col = getattr(Job, field_name)

                if op_suffix == "eq":
                    query = query.filter(col == value)
                elif op_suffix == "lte":
                    query = query.filter(col <= value)
                elif op_suffix == "lt":
                    query = query.filter(col < value)
                elif op_suffix == "gte":
                    query = query.filter(col >= value)
                elif op_suffix == "gt":
                    query = query.filter(col > value)
                elif op_suffix == "ne":
                    query = query.filter(col != value)
                else:
                    raise ValueError(
                        f"Unsupported or missing operator suffix for field '{field_name}'. "
                        f"Allowed: __eq, __lte, __lt, __gte, __gt, __ne"
                    )

            query = query.options(
                joinedload(getattr(Job, "routing"))
                .joinedload(getattr(Routing, "operations"))
            ).order_by(Job.arrival)

            jobs = query.all()
            session.expunge_all()
            return list(jobs)


    @classmethod
    def get_by_source_name_max_util_and_lt_arrival(
            cls, source_name: str, max_bottleneck_utilization: Decimal, arrival_limit: int) -> List[Job]:
        """
        Retrieves all jobs with the given RoutingSource,
        whose max_bottleneck_utilization == value AND arrival < limit.
        """
        job_filters = {
            "max_bottleneck_utilization__eq": max_bottleneck_utilization,
            "arrival__lt": arrival_limit
        }
        return cls._get_by_source_name_and_job_filters(
            source_name=source_name,
            job_filters=job_filters
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


# ExperimentQuery ---------------------------------------------------------------------------------
class ExperimentQuery:
    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated.")

    @staticmethod
    def get_experiment_only(experiment_id: int) -> Experiment:
        """
        Retrieve a single :class:`Experiment` by its primary key.

        :param experiment_id: Primary key of the experiment to fetch.
        :returns: The matching :class:`Experiment` instance.
        """
        with SessionLocal() as session:
            exp = (
                session.query(Experiment)
                .filter(Experiment.id == experiment_id)
                .one_or_none()
            )

            if exp is None:
                raise ValueError(f"Experiment with id={experiment_id} not found.")

            session.expunge(exp)
            return exp

    @staticmethod
    def save_schedule_jobs(experiment_id: int, shift_number: int, live_jobs: Iterable[LiveJob]):
        """
        Build ScheduleJob and ScheduleOperation ORM objects.

        This keeps the relationship to ScheduleJob view-only, so ScheduleOperation
        must be created and tracked separately. No Session is used here — objects
        are returned detached and can be added later.

        :param experiment_id: ID of the experiment the jobs belong to.
        :param shift_number: Shift number for all schedule jobs.
        :param live_jobs: Iterable of LiveJob dataclasses (source data).
        :return: (List of ScheduleJob, List of ScheduleOperation), both not yet persisted.
        """
        schedule_jobs: list[ScheduleJob] = []
        schedule_operations: list[ScheduleOperation] = []

        for lj in live_jobs:
            # Create ScheduleJob purely in memory
            sj = ScheduleJob(
                id=lj.id,  # PK matching the job.id
                experiment_id=experiment_id,
                shift_number=shift_number
            )
            schedule_jobs.append(sj)

            # Create ScheduleOperation objects separately (since relationship is viewonly)
            for op in lj.operations:
                so = ScheduleOperation(
                    job_id=lj.id,
                    experiment_id=experiment_id,  # must be set manually
                    shift_number=shift_number,  # must be set manually
                    position_number=op.position_number,
                    start=op.start,
                    end=op.end,
                )
                schedule_operations.append(so)

        with SessionLocal() as session:
            session.add_all(schedule_jobs + schedule_operations)
            session.commit()


    @staticmethod
    def save_simulation_jobs(experiment_id: int, live_jobs: Iterable[LiveJob]):
        """
        Build SimulationJob and SimulationOperation ORM objects.

        Relationships are view-only, so SimulationOperation must be collected separately.
        No Session is used here — returned objects are detached and can be added later.

        :param experiment_id: ID of the experiment the simulation belongs to.
        :param live_jobs: Iterable of LiveJob dataclasses (source data).
        :return: (List[SimulationJob], List[SimulationOperation]), both not yet persisted.
        """
        sim_jobs: List[SimulationJob] = []
        sim_ops: List[SimulationOperation] = []

        for lj in live_jobs:
            # Parent row
            sj = SimulationJob(
                id=lj.id,  # PK matching job.id
                experiment_id=experiment_id,
            )
            sim_jobs.append(sj)

            # Child rows (no shift_number here; include duration)
            for op in lj.operations:
                so = SimulationOperation(
                    job_id=lj.id,
                    experiment_id=experiment_id,
                    position_number=op.position_number,
                    start=op.start,
                    duration=op.duration,  # <-- wichtig für SimulationOperation
                    end=op.end,
                )
                sim_ops.append(so)

        with SessionLocal() as session:
            session.add_all(sim_jobs + sim_ops)
            session.commit()

