from typing import List, final, Union

from sqlalchemy import and_, or_
from sqlalchemy.orm import joinedload, InstrumentedAttribute, selectinload
from omega.db_setup import SessionLocal
from omega.db_models import Job, Routing, JobOperation, RoutingSource

@final
class RoutingQuery:
    def __new__(cls, *args, **kwargs):
        raise TypeError("RoutingQuery is a static utility class and cannot be instantiated.")

    @staticmethod
    def get_all() -> list[Routing]:
        with SessionLocal() as session:
            routings = (
                session.query(Routing)
                .options(
                    joinedload(getattr(Routing, "routing_source")),
                    joinedload(getattr(Routing, "operations"))
                )
                .all()
            )
            session.expunge_all()
            return routings

    @staticmethod
    def get_by_source_name(source_name: str) -> list[Routing]:
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
            return routings


class JobQuery:

    @staticmethod
    def _get_by_field(field_name: str, field_value: Union[str, int]) -> List[Job]:
        if field_name not in Job.__mapper__.columns.keys():     # type: ignore[attr-defined]
            raise ValueError(f"Field '{field_name}' is not a valid column in Job.")

        with SessionLocal() as session:
            query = session.query(Job).options(
                joinedload(getattr(Job, "routing")),
                joinedload(getattr(Job, "operations")).joinedload(getattr(JobOperation, "routing_operation")),
                joinedload(getattr(Job, "experiment"))
            )
            jobs = query.filter(getattr(Job, field_name) == field_value).all()
            session.expunge_all()
            return jobs

    @classmethod
    def get_by_routing_id(cls, routing_id: str) -> List[Job]:
        return cls._get_by_field("routing_id", routing_id)

    @classmethod
    def get_by_experiment_id(cls, experiment_id: int) -> List[Job]:
        return cls._get_by_field("experiment_id", experiment_id)


    @staticmethod
    def get_by_earliest_start(experiment_id: int, earliest_start: int) -> List[Job]:
        with SessionLocal() as session:
            jobs = session.query(Job).filter(
                Job.earliest_start == earliest_start, Job.experiment_id == experiment_id).options(
                joinedload(getattr(Job, "routing")),
                joinedload(getattr(Job, "operations")).joinedload(getattr(JobOperation, "routing_operation"))
                ).all()
            session.expunge_all()
            return jobs

    @staticmethod
    def get_by_earliest_start_or_ids(
            experiment_id: int,earliest_start: int,
            job_ids: List[int]) -> List[Job]:
        """
        Retrieve all jobs for a given experiment where either the earliest start time matches
        the specified value or the job ID is in the given list.

        :param experiment_id: ID of the experiment to filter jobs by.
        :param earliest_start: Earliest start time to match.
        :param job_ids: List of job IDs to include in the result.
        :return: List of Job instances with routing and operation details loaded.
        """

        with SessionLocal() as session:

            job_id_attr: InstrumentedAttribute = getattr(Job, "id")

            jobs = session.query(Job).filter(
                and_(
                    Job.experiment_id == experiment_id,
                    or_(
                        Job.earliest_start == earliest_start,
                        job_id_attr.in_(job_ids)
                    )
                )
            ).options(
                joinedload(getattr(Job, "routing")),
                joinedload(getattr(Job, "operations")).joinedload(getattr(JobOperation, "routing_operation"))
            ).all()
            session.expunge_all()
            return jobs




