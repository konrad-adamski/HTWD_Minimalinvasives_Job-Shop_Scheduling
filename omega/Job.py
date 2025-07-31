from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
from sqlalchemy import Column, Integer, String, ForeignKey, ForeignKeyConstraint
from sqlalchemy.orm import relationship

from omega.Routing import mapper_registry, Routing, RoutingOperation


# von dir definierte Klassen:
# from your_module import Routing, JobOperation

@mapper_registry.mapped
@dataclass
class Job:
    __tablename__ = "job"
    __sa_dataclass_metadata_key__ = "sa"

    # Technischer Primärschlüssel
    id: int = field(init=False, default=None, metadata={
        "sa": Column(Integer, primary_key=True, autoincrement=True)
    })

    # Domänenschlüssel
    job_id: str = field(default="", metadata={
        "sa": Column(String(255), nullable=False)
    })

    routing_id: Optional[int] = field(default=None, metadata={
        "sa": Column(Integer, ForeignKey("routing.id"), nullable=False)
    })


    # Zeitinformationen
    arrival: int = field(default=0, metadata={
        "sa": Column(Integer, nullable=False)
    })

    earliest_start: int = field(default=0, metadata={
        "sa": Column(Integer, nullable=False)
    })

    deadline: int = field(default=0, metadata={
        "sa": Column(Integer, nullable=False)
    })

    # Relations
    routing: Optional[Routing] = field(default=None, repr=False, metadata={
        "sa": relationship(
            "Routing",
            back_populates="jobs"
        )
    })

    # Beziehung zu JobOperationen
    operations: List[JobOperation] = field(default_factory=list, repr=False, metadata={
        "sa": relationship(
            "JobOperation",
            back_populates="job",
            cascade="all, delete-orphan"
        )
    })

    @mapper_registry.mapped
    @dataclass
    class JobOperation:
        __tablename__ = "job_operation"
        __sa_dataclass_metadata_key__ = "sa"

        id: int = field(init=False, default=None, metadata={
            "sa": Column(Integer, primary_key=True, autoincrement=True)
        })

        job_id: str = field(metadata={
            "sa": Column(String, nullable=False)
        })

        position_number: int = field(metadata={
            "sa": Column(Integer, nullable=False)
        })

        routing_id: str = field(metadata={
            "sa": Column(String, nullable=False)
        })

        # Beziehung zur RoutingOperation via (routing_id, position_number)
        routing_operation: RoutingOperation = field(metadata={
            "sa": relationship(
                "RoutingOperation",
                primaryjoin=(
                    "and_(JobOperation.routing_id == RoutingOperation.routing_id, "
                    "JobOperation.position_number == RoutingOperation.position_number)"
                ),
                viewonly=True,
                lazy="joined"
            )
        })

        @property
        def machine(self) -> str:
            return self.routing_operation.machine

        @property
        def duration(self) -> int:
            return self.routing_operation.duration