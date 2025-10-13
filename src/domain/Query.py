from __future__ import annotations

import numpy as np
import pandas as pd

from decimal import Decimal
from typing import List, Union, Iterable, Tuple, Optional

from sqlalchemy import text, create_engine
from sqlalchemy.orm import joinedload, sessionmaker

from src.domain.orm_models import Routing, RoutingSource, Job, Machine, Experiment, ScheduleOperation, ScheduleJob, \
    LiveJob, SimulationJob, SimulationOperation, RoutingOperation, MachineInstance
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
                    joinedload(getattr(Routing, "operations")).joinedload(getattr(RoutingOperation, "machine"))
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
        cls, source_name: str, field_name: Optional[str] = None,
        field_value: Optional[Union[str, int, Decimal]] = None) -> List[Job]:
        """
        Retrieve jobs filtered by a required RoutingSource name and (optionally) an additional Job field.

        :param source_name: Name of the RoutingSource (via Job.routing.routing_source.name).
        :param field_name: (Optional) Name of a Job column to filter on.
        :param field_value: (Optional) Value for the Job column filter.
        :return: List of Job instances with routing and operations eagerly loaded.
        """

        with SessionLocal() as session:
            query = (
                session.query(Job)
                .join(Job.routing)                 # Job -> Routing
                .join(Routing.routing_source)      # Routing -> RoutingSource
                .filter(RoutingSource.name == source_name)
                .options(
                    joinedload(getattr(Job, "routing"))
                    .joinedload(getattr(Routing, "operations")).joinedload(getattr(RoutingOperation, "machine"))
                )
                .order_by(Job.arrival)
            )

            # Zusätzlichen Filter nur anwenden, wenn beide Parameter vorhanden sind
            if field_name is not None and field_value is not None:
                # Validieren nur, wenn wir den Filter tatsächlich anwenden
                if field_name not in Job.__mapper__.columns.keys():  # type: ignore[attr-defined]
                    raise ValueError(f"Field '{field_name}' is not a valid column in Job.")
                query = query.filter(getattr(Job, field_name) == field_value)

            jobs = query.all()
            session.expunge_all()
            return list(jobs)

    @classmethod
    def get_by_source_name(cls, source_name: str) -> List[Job]:
        """
        Retrieve all jobs for a given RoutingSource name.

        :param source_name: Name der RoutingSource.
        """
        return cls._get_by_source_name_and_field(
            source_name=source_name,

        )

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
    def get_by_source_name(source_name: str) -> list[Machine]:
        """
        Retrieve all machines for a given routing source,
        with the RoutingSource eagerly loaded.

        :param source_name: Name of the routing source.
        :return: List of Machine instances.
        """
        with SessionLocal() as session:
            machines = (
                session.query(Machine)
                .join(Machine.source)
                .filter(
                    RoutingSource.name == source_name
                )
                #.options(joinedload(getattr(Machine, "source")))
                .order_by(Machine.name)
                .all()
            )

            session.expunge_all()
            return list(machines)

class MachineInstanceQuery:
    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated.")

    @staticmethod
    def get_by_source_name_and_max_bottleneck_utilization(source_name: str, max_bottleneck_utilization: Decimal) -> list[MachineInstance]:
        """
        Retrieve all MachineInstance rows for a given RoutingSource.name and max_bottleneck_utilization,
        with Machine and RoutingSource eagerly loaded.
        """
        with SessionLocal() as session:
            machine_instances = (
                session.query(MachineInstance)
                .join(MachineInstance.machine)
                .join(Machine.source)
                .filter(
                    RoutingSource.name == source_name,
                    MachineInstance.max_bottleneck_utilization == max_bottleneck_utilization
                )
                .options(
                    joinedload(getattr(MachineInstance, "machine")).joinedload(getattr(Machine, "source"))
                )
                .order_by(Machine.name)
                .all()
            )

            session.expunge_all()
            return machine_instances

# ExperimentQuery ---------------------------------------------------------------------------------
class ExperimentQuery:
    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated.")

    @staticmethod
    def get_experiment(experiment_id: int) -> Experiment:
        """
        Retrieve a single Experiment by its primary key, with required relations eagerly loaded
        so it can be safely accessed after the session is closed.
        """
        with SessionLocal() as session:
            exp = (
                session.query(Experiment)
                .options(
                    joinedload(getattr(Experiment, "routing_source"))
                )
                .filter(Experiment.id == experiment_id)
                .one_or_none()
            )

            if exp is None:
                raise ValueError(f"Experiment with id={experiment_id} not found.")

            # Detach the fully loaded object so it can be used outside the session
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


class ExperimentAnalysisQuery:
    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated.")

    @staticmethod
    def get_experiments(
            experiment_id: Optional[int] = None,
            max_bottleneck_utilization: Optional[float] = None,
            db_path: Optional[str] = None,
    ) -> List[Experiment]:
        """
        Liefert eine Liste von Experiment-Objekten.
        - experiment_id=None → alle Experimente
        - max_bottleneck_utilization: optionaler Filter auf Experiment.max_bottleneck_utilization
        """
        if db_path:
            my_engine = create_engine(f"sqlite:///{db_path}")
            SessionFactory = sessionmaker(bind=my_engine)
        else:
            SessionFactory = SessionLocal
        with SessionFactory() as session:
            query = session.query(Experiment)

            if experiment_id is not None:
                query = query.filter(Experiment.id == experiment_id)

            if max_bottleneck_utilization is not None:
                query = query.filter(
                    Experiment.max_bottleneck_utilization == max_bottleneck_utilization
                )

            experiments = query.order_by(Experiment.id).all()
            session.expunge_all()
            return experiments

    @staticmethod
    def get_schedule_jobs(
        experiment_id: Optional[int] = None,
        max_bottleneck_utilization: Optional[float] = None,
        db_path: Optional[str] = None
    ) -> List[ScheduleJob]:
        """
        Liefert alle ScheduleJobs (inkl. Operations, Job->Routing, Experiment).
        - experiment_id=None → alle Experimente
        - max_bottleneck_utilization: optionaler Filter auf Experiment.max_bottleneck_utilization (exakte Übereinstimmung)
        """
        if db_path:
            my_engine = create_engine(f"sqlite:///{db_path}")
            SessionFactory = sessionmaker(bind=my_engine)
        else:
            SessionFactory = SessionLocal
        with SessionFactory() as session:
            query = (
                session.query(ScheduleJob)
                .options(
                    joinedload(getattr(ScheduleJob, "operations")),
                    joinedload(getattr(ScheduleJob, "job"))
                    .joinedload(getattr(Job, "routing"))
                    .joinedload(getattr(Routing, "operations")),
                    joinedload(getattr(ScheduleJob, "experiment")),
                )
            )

            if experiment_id is not None:
                query = query.filter(ScheduleJob.experiment_id == experiment_id)

            if max_bottleneck_utilization is not None:
                # Join nur dann, wenn wir wirklich danach filtern
                query = query.join(getattr(ScheduleJob, "experiment")).filter(
                    Experiment.max_bottleneck_utilization == max_bottleneck_utilization
                )

            jobs = query.order_by(ScheduleJob.shift_number, ScheduleJob.id).all()
            session.expunge_all()
            return jobs

    @classmethod
    def get_experiments_dataframe(
            cls,
            experiment_id: Optional[int] = None,
            max_bottleneck_utilization: Optional[float] = None,
            id_column: str = "Experiment_ID",
            lateness_ratio_column: str = "Abs Lateness Ratio",
            tardiness_ratio_column: str = "Inner Tardiness Ratio",
            bottleneck_column: str = "Max Bottleneck Utilization",
            sim_sigma_column: str = "Sim Sigma",
            shift_length_column: str = "Shift Length",
            w_t_column: str = "w_t",
            w_e_column: str = "w_e",
            w_dev_column: str = "w_dev",
            experiment_type_column: str = "Experiment_Type",
            db_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Baut ein DataFrame aus Experimenten:
        - Basisparameter (id, source_id, ratios, bottleneck, sim_sigma, shift_length)
        - Berechnete Gewichte (w_t, w_e, w_dev) aus get_solver_weights()
        Spaltennamen sind frei parametrisierbar.
        """
        exps = cls.get_experiments(
            experiment_id=experiment_id,
            max_bottleneck_utilization=max_bottleneck_utilization,
            db_path=db_path
        )

        records = []
        for e in exps:
            w_t, w_e, w_dev = e.get_solver_weights()
            records.append({
                id_column: e.id,
                lateness_ratio_column: float(e.absolute_lateness_ratio),
                tardiness_ratio_column: float(e.inner_tardiness_ratio),
                bottleneck_column: float(e.max_bottleneck_utilization),
                sim_sigma_column: float(e.sim_sigma),
                shift_length_column: int(e.shift_length),
                w_t_column: int(w_t),
                w_e_column: int(w_e),
                w_dev_column: int(w_dev),
                experiment_type_column: e.type
            })

        ordered_cols = [
            id_column,
            lateness_ratio_column,
            tardiness_ratio_column,
            bottleneck_column,
            sim_sigma_column,
            shift_length_column,
            w_t_column,
            w_e_column,
            w_dev_column,
            experiment_type_column
        ]
        df = pd.DataFrame.from_records(records, columns=ordered_cols)
        return df.sort_values([id_column], ignore_index=True)
    
    @classmethod
    def get_schedule_jobs_operations_dataframe(
        cls,
        experiment_id: Optional[int] = None,
        max_bottleneck_utilization: Optional[float] = None,
        job_column: str = "Job",
        routing_column: str = "Routing_ID",
        experiment_column: str = "Experiment_ID",
        shift_column: str = "Shift",
        position_column: str = "Operation",
        machine_column: str = "Machine",
        og_duration_column: str = "Original Duration",
        start_column: str = "Start",
        end_column: str = "End",
        arrival_column: str = "Arrival",
        earliest_start_column = "Ready Time",
        due_date_column: str = "Due Date",
        db_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Baut ein DataFrame aus ScheduleJobs.
        experiment_id=None → keine Filterung, es werden alle zurückgegeben.
        """
        jobs = cls.get_schedule_jobs(experiment_id, max_bottleneck_utilization, db_path=db_path)

        records = []
        for sj in jobs:
            for op in sj.operations:
                route_op = sj.job.routing.get_operation_by_position(op.position_number)
                records.append({
                    job_column: sj.id,
                    routing_column: sj.job.routing_id,
                    experiment_column: sj.experiment_id,
                    shift_column: sj.shift_number,
                    position_column: op.position_number,
                    machine_column: route_op.machine_name,
                    arrival_column: sj.job.arrival,
                    earliest_start_column: sj.job.earliest_start,
                    due_date_column: sj.job.due_date,
                    og_duration_column: route_op.duration,
                    start_column: op.start,
                    end_column: op.end,
                })

        ordered_cols = [
            job_column,
            routing_column,
            experiment_column,
            arrival_column,
            earliest_start_column,
            due_date_column,
            shift_column,
            position_column,
            machine_column,
            og_duration_column,
            start_column,
            end_column,
        ]

        df = (
            pd.DataFrame.from_records(records, columns=ordered_cols)
            .sort_values([shift_column, job_column, position_column], ignore_index=True)
        )
        return df