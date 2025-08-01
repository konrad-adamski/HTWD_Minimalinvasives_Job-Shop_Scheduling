from typing import List
from sqlalchemy.orm import joinedload
from db_setup import SessionLocal
from omega.db_models import Job


class JobQueryHelper:

    @staticmethod
    def get_by_routing(routing_id: str) -> List[Job]:
        with SessionLocal() as session:
            return session.query(Job).filter_by(routing_id=routing_id).options(
                joinedload(getattr(Job, "routing")),
                joinedload(getattr(Job, "operations"))
                ).all()

    @staticmethod
    def get_by_earliest_start(value: int = None) -> List[Job]:
        with SessionLocal() as session:
            return session.query(Job).filter(Job.earliest_start == value).options(
                joinedload(getattr(Job, "routing")),
                joinedload(getattr(Job, "operations"))
            ).all()

    @staticmethod
    def get_all() -> List[Job]:
        with SessionLocal() as session:
            return  session.query(Job).options(
                    joinedload(getattr(Job, "routing")),
                    joinedload(getattr(Job, "operations"))
                ).all()
