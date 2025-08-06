from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from sqlalchemy import Column, Integer, String, ForeignKey, ForeignKeyConstraint, Float, and_
from sqlalchemy.orm import relationship
from typing import Optional, List, Union

from src.classes.orm_setup import mapper_registry


@mapper_registry.mapped
@dataclass
class RoutingSource:
    __tablename__ = "routing_source"
    __sa_dataclass_metadata_key__ = "sa"

    id: Optional[int] = field(default=None, init=False, metadata={
        "sa": Column(Integer, primary_key=True, autoincrement=True)
    })
    name: str = field(default="Unknown Routing Set", metadata={
        "sa": Column(String(255), nullable=False, unique=True, default="Unknown Routing Set")
    })

    # Routings, die zu dieser Instanz gehören
    routings: List[Routing] = field(
        default_factory=list,
        repr=False,
        metadata={
            "sa": relationship(
                "Routing",
                back_populates="routing_source",
                cascade="all, delete-orphan"
            )
        }
    )

@mapper_registry.mapped
@dataclass
class Machine:
    __tablename__ = "machine"
    __sa_dataclass_metadata_key__ = "sa"

    name: str = field(metadata={
        "sa": Column(String(100), primary_key=True)  # Maschinenname als eindeutige ID
    })

    # Optional: Beziehung zu RoutingOperations
    operations: List[RoutingOperation] = field(default_factory=list, repr=False, metadata={
        "sa": relationship(
            "RoutingOperation",
            back_populates="machine"
        )
    })

    utilization_machines: List[UtilizationMachine] = field(default_factory=list, repr=False, metadata={
        "sa": relationship("UtilizationMachine", back_populates="machine")
    })

    def get_transition_by_max_bottleneck_utilization(self, max_bottleneck_utilization: float) -> Optional[int]:
        """Gibt die transition_time für das gegebene Experiment zurück, falls vorhanden."""
        for em in self.utilization_machines:
            if em.max_bottleneck_utilization == max_bottleneck_utilization:
                return em.transition_time
        return 0

@mapper_registry.mapped
@dataclass
class UtilizationMachine:
    __tablename__ = "utilization_machine"
    __sa_dataclass_metadata_key__ = "sa"

    name: str = field(init=False, metadata={
    "sa": Column(String(100), ForeignKey("machine.name"), primary_key=True)
    })


    max_bottleneck_utilization: float = field(default=0.5, metadata={
            "sa": Column(Float, primary_key=True)
    })

    transition_time: int = field(default=0, metadata={
        "sa": Column(Integer, nullable=False)
    })

    machine: Machine = field(default=None, repr=False, metadata={
        "sa": relationship("Machine", back_populates="utilization_machines")
    })

    def __post_init__(self):
        if self.machine:
            self.name = self.machine.name



@mapper_registry.mapped
@dataclass
class Routing:
    __tablename__ = "routing"
    __sa_dataclass_metadata_key__ = "sa"

    id: str = field(metadata={
        "sa": Column(String(255), primary_key=True)
    })

    source_id: Optional[int] = field(default=None, metadata={
        "sa": Column(Integer, ForeignKey("routing_source.id"), nullable=True)
    })

    routing_source: Optional[RoutingSource] = field(
        default=None,
        repr=False,
        metadata={
            "sa": relationship("RoutingSource", back_populates="routings", lazy="joined")
        }
    )

    operations: List[RoutingOperation] = field(default_factory=list, repr=False, metadata={"sa": relationship(
                "RoutingOperation", back_populates="routing",
                cascade="all, delete-orphan", lazy="joined")
    })

    jobs: List[Job] = field(default_factory=list, repr=False, metadata={"sa": relationship(
                "Job", back_populates="routing",
                cascade="all, delete-orphan", lazy="joined")
    })

    @property
    def source_name(self) -> str:
        return self.routing_source.name if self.routing_source is not None else None

    # Custom-Property für die Planungslogik
    @property
    def sum_duration(self) -> int:
        """
        Get the total duration of all operations in this routing.

        :return: Sum of durations of all operations
        """
        return sum(op.duration for op in self.operations)

    def sum_left_duration(self, position: int) -> int:
        return sum(op.duration for op in self.operations if op.position_number >= position)

    def get_operation_by_position(self, position: int) -> RoutingOperation:
        for op in self.operations:
            if op.position_number == position:
                return op
        raise ValueError(f"No operation with position_number={position} in routing {self.id}")



@mapper_registry.mapped
@dataclass
class RoutingOperation:
    __tablename__ = "routing_operation"
    __sa_dataclass_metadata_key__ = "sa"

    routing_id: str = field(metadata={
        "sa": Column(String(255), ForeignKey("routing.id"), primary_key=True)
    })

    position_number: int = field(metadata={
        "sa": Column(Integer, primary_key=True)
    })

    machine_name: str = field(init=True, metadata={
        "sa": Column(String(100), ForeignKey("machine.name"), nullable=False)
    })

    duration: int = field(metadata={
        "sa": Column(Integer, nullable=False)
    })


    routing: Routing = field(
        default=None,
        repr=False,
        metadata={
            "sa": relationship("Routing", back_populates="operations")
        }
    )

    machine: Optional[Machine] = field(init=False, default=None, repr=False, metadata={
        "sa": relationship("Machine", back_populates="operations")
    })


@mapper_registry.mapped
@dataclass
class Job:
    __tablename__ = "job"
    __sa_dataclass_metadata_key__ = "sa"

    def __repr__(self) -> str:
        attrs = {
            "id": self.id,
            "routing_id": self.routing_id,
            "experiment_id": self.experiment_id,
            "arrival": self.arrival,
            "earliest_start": self.earliest_start,
            "deadline": self.deadline,
            "sum_duration": self.sum_duration,
        }
        return "Job(" + ", ".join(f"{key}={value!r}" for key, value in attrs.items()) + ")"

    id: str = field(default="", metadata={
        "sa": Column(String(255), nullable=False, primary_key=True)
    })

    routing_id: str = field(init=False, default="", metadata={
        "sa": Column(String(255), ForeignKey("routing.id"), nullable=False)
    })

    experiment_id: int = field(init=False, default=None, metadata={
        "sa": Column(Integer, ForeignKey("experiment.id"), nullable=False)
    })

    arrival: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})

    deadline: Optional[int] = field(default=None, metadata={"sa": Column(Integer)})

    experiment: Experiment = field(default=None, repr=False, metadata={
        "sa": relationship("Experiment", back_populates="jobs", lazy="joined")
    })

    routing: Routing = field(default=None, repr=False, metadata={
        "sa": relationship("Routing", back_populates="jobs", lazy="joined")
    })

    schedule_jobs: List[ScheduleJob] = field(default_factory=list, repr=False, metadata={
        "sa": relationship("ScheduleJob", back_populates="job", cascade="all, delete-orphan", lazy="joined")
    })

    simulation_job: Optional[SimulationJob] = field(default=None, metadata={
        "sa": relationship("SimulationJob", back_populates="job", uselist=False, lazy="joined")
    })


    # Kein ORM-Relationship mehr zu JobOperation – stattdessen dynamisch generiert
    @property
    def operations(self) -> List[JobOperation]:
        operations: List[JobOperation] = []
        for routing_op in self.routing.operations:
            operations.append(JobOperation(
                job=self,
                position_number=routing_op.position_number,
                machine_name=routing_op.machine_name,
                duration=routing_op.duration,
            ))
        return operations

    @property
    def earliest_start(self) -> int:
        return int(np.ceil((self.arrival + 1) / 1440) * 1440)

    @property
    def max_bottleneck_utilization(self) -> float:
        return float(self.experiment.max_bottleneck_utilization)

    @property
    def sum_duration(self) -> int:
        return self.routing.sum_duration if self.routing else 0

    @property
    def last_operation_position_number(self) -> Optional[int]:
        """
        Returns the highest position_number among all operations,
        i.e., the last technological step of the job.
        """
        if not self.operations:
            return None
        return max(op.position_number for op in self.operations)

    def __post_init__(self):
        if self.routing:
            self.routing_id = self.routing.id
        if self.experiment:
            self.experiment_id = self.experiment.id


@mapper_registry.mapped
@dataclass
class SimulationJob:
    __tablename__ = "simulation_job"
    __sa_dataclass_metadata_key__ = "sa"

    id: str = field(metadata={
        "sa": Column(String, ForeignKey("job.id"), primary_key=True)
    })

    job: Job = field(default=None, repr=False, metadata={
        "sa": relationship("Job", back_populates="simulation_job", lazy="joined")
    })

    operations: List[SimulationOperation] = field(default_factory=list, metadata={
        "sa": relationship(
            "SimulationOperation",
            back_populates="simulation_job",
            cascade="all, delete-orphan",
            lazy="joined"
        )
    })

    @property
    def experiment_id(self) -> int:
        return self.job.experiment_id

    @property
    def routing(self) -> Routing:
        return self.job.routing

    @property
    def routing_id(self) -> str:
        return self.job.routing_id

    @property
    def arrival(self) -> int:
        return self.job.arrival

    @property
    def earliest_start(self) -> int:
        return self.job.earliest_start

    @property
    def deadline(self) -> int:
        return self.job.deadline

    @property
    def last_operation_position_number(self) -> Optional[int]:
        """
        Returns the highest position_number among all operations,
        i.e., the last technological step of the job.
        """
        if not self.operations:
            return None
        return max(op.position_number for op in self.operations)




@mapper_registry.mapped
@dataclass
class ScheduleJob:
    __tablename__ = "schedule_job"
    __sa_dataclass_metadata_key__ = "sa"

    id: str = field(metadata={
        "sa": Column(String, ForeignKey("job.id"), primary_key=True)
    })

    shift_number: int = field(metadata={
        "sa": Column(Integer, primary_key=True)
    })

    experiment_id: int = field(metadata={
        "sa": Column(Integer, ForeignKey("experiment.id"), nullable=False)
    })

    # Beziehungen zu Job und Experiment
    job: Job = field(default=None, repr=False, metadata={
        "sa": relationship("Job", back_populates="schedule_jobs", lazy="joined")
    })


    # Neue Beziehung zu Shift
    shift: Shift = field(default=None, repr=False, metadata={
        "sa": relationship(
            "Shift",
            primaryjoin=(
                "and_(ScheduleJob.experiment_id == Shift.experiment_id, "
                "ScheduleJob.shift_number == Shift.shift_number)"
            ),
            back_populates="schedule_jobs",
            lazy="joined"
        )
    })

    operations: List[ScheduleOperation] = field(default_factory=list, metadata={
        "sa": relationship(
            "ScheduleOperation",
            back_populates="schedule_job",
            cascade="all, delete-orphan",
            lazy="joined"
        )
    })

    @property
    def routing(self) -> Routing:
        return self.job.routing

    @property
    def routing_id(self) -> str:
        return self.job.routing_id

    @property
    def arrival(self) -> int:
        return self.job.arrival

    @property
    def earliest_start(self) -> int:
        return self.job.earliest_start

    @property
    def deadline(self) -> int:
        return self.job.deadline

    @property
    def last_operation_position_number(self) -> Optional[int]:
        """
        Returns the highest position_number among all operations,
        i.e., the last technological step of the job.
        """
        if not self.operations:
            return None
        return max(op.position_number for op in self.operations)

    __table_args__ = (
        ForeignKeyConstraint(
            ["experiment_id", "shift_number"],
            ["shift.experiment_id", "shift.shift_number"]
        ),
    )


@mapper_registry.mapped
@dataclass
class Experiment:
    __tablename__ = "experiment"
    __sa_dataclass_metadata_key__ = "sa"

    id: int = field(init=False, metadata={
        "sa": Column(Integer, primary_key=True, autoincrement=True)
    })

    total_shift_number: int = field(init=True, default=None, metadata={
        "sa": Column(Integer, nullable=False)
    })

    main_pct: float = field(default=0.5, metadata={
        "sa": Column(Float, nullable=False)
    })

    w_t: int = field(default=1, metadata={
        "sa": Column(Integer, nullable=False)
    })

    w_e: int = field(default=1, metadata={
        "sa": Column(Integer, nullable=False)
    })

    w_first: Optional[int] = field(default=1, metadata={
        "sa": Column(Integer, nullable=True)
    })

    max_bottleneck_utilization: float = field(default=0.5, metadata={
        "sa": Column(Float, nullable=False)
    })

    sim_sigma: float = field(default=0.0, metadata={
        "sa": Column(Float, nullable=False)
    })

    jobs: List[Job] = field(default_factory=list, repr=False, metadata={
        "sa": relationship("Job", back_populates="experiment", cascade="all, delete-orphan", lazy="joined")
    })

    shifts: List[Shift] = field(default_factory=list, repr=False, metadata={"sa": relationship(
        "Shift", back_populates="experiment", cascade="all, delete-orphan", lazy="joined")
    })

    def __post_init__(self):
        if not (0 <= self.max_bottleneck_utilization <= 1):
            raise ValueError("max_bottleneck_utilization must be between 0 and 1.")
        if not (0 <= self.main_pct <= 1):
            raise ValueError("main_pct must be between 0 and 1.")


@mapper_registry.mapped
@dataclass
class Shift:
    __tablename__ = "shift"
    __sa_dataclass_metadata_key__ = "sa"

    shift_number: int = field(metadata={
        "sa": Column(Integer, primary_key=True)
    })

    experiment_id: int = field(metadata={
        "sa": Column(Integer, ForeignKey("experiment.id"), primary_key=True)
    })

    shift_length: int = field(init=False, default=1440, metadata={
        "sa": Column(Integer, nullable=False)
    })

    experiment: Experiment = field(default=None, repr=False, metadata={
        "sa": relationship(
            Experiment,
            back_populates="shifts",
            lazy="joined"
        )
    })

    schedule_jobs: List[ScheduleJob] = field(default_factory=list, repr=False, metadata={
        "sa": relationship(
            ScheduleJob,
            back_populates="shift",
            cascade="all, delete-orphan",
            lazy="joined"
        )
    })

    @property
    def shift_start(self) -> int:
        return self.shift_number * self.shift_length

    @property
    def shift_end(self) -> int:
        return self.shift_start + self.shift_length

    def __post_init__(self):
        if self.shift_length is None:
            self.shift_length = 1440

@mapper_registry.mapped
@dataclass
class ScheduleOperation:
    __tablename__ = "schedule_operation"
    __sa_dataclass_metadata_key__ = "sa"

    job_id: str = field(metadata={
        "sa": Column(String, ForeignKey("schedule_job.id"), primary_key=True)
    })

    position_number: int = field(metadata={
        "sa": Column(Integer, primary_key=True)
    })

    start: int = field(default=0, metadata={
        "sa": Column(Integer, nullable=False)
    })

    end: int = field(default=0, metadata={
        "sa": Column(Integer, nullable=False)
    })

    schedule_job: ScheduleJob = field(default=None, repr=False, metadata={
        "sa": relationship(
            ScheduleJob,
            back_populates="operations",
            lazy="joined"
        )
    })


    @property
    def _routing_operation(self) -> RoutingOperation:
        return self.schedule_job.job.routing.get_operation_by_position(self.position_number)

    @property
    def machine_name(self) -> str:
        return self._routing_operation.machine_name

    @property
    def duration(self) -> int:
        return self._routing_operation.duration


@mapper_registry.mapped
@dataclass
class SimulationOperation:
    __tablename__ = "simulation_operation"
    __sa_dataclass_metadata_key__ = "sa"


    job_id: str = field(metadata={
        "sa": Column(String, ForeignKey("simulation_job.id"), primary_key=True)
    })

    position_number: int = field(metadata={"sa": Column(Integer, primary_key=True)})

    start: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})

    duration: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})

    end: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})

    simulation_job: SimulationJob = field(default=None, repr=False, metadata={"sa": relationship(
        SimulationJob, back_populates="operations", lazy="joined")
    })

    @property
    def _routing_operation(self) -> RoutingOperation:
        return self.simulation_job.job.routing.get_operation_by_position(self.position_number)

    @property
    def machine_name(self) -> str:
        return self._routing_operation.machine_name

    @property
    def route_duration(self) -> int:
        return self._routing_operation.duration



# ---------------------------------------------------------------------------------------------------------------------
# View/Helper classes (not ORM models): wrap ORM objects for easy access.

@dataclass
class JobTemplate:
    id: str
    routing_id: Optional[str] = None
    experiment_id: Optional[int] = None
    arrival: Optional[int] = None
    deadline: Optional[int] = None
    max_bottleneck_utilization: Optional[float] = None

    operations: List[JobOperation] = field(default_factory=list)

    @property
    def earliest_start(self) -> int:
        if self.arrival is None:
            return 0
        return int(np.ceil((self.arrival + 1) / 1440) * 1440)


    # Custom-Property für die Planungslogik
    @property
    def sum_duration(self) -> int:
        """
        Get the total duration of all operations for this job.

        :return: Sum of durations of all operations
        """
        return sum(op.duration for op in self.operations)

    @property
    def last_operation_position_number(self) -> Optional[int]:
        """
        Returns the highest position_number among all operations,
        i.e., the last technological step of the job.
        """
        if not self.operations:
            return None
        return max(op.position_number for op in self.operations)

    @classmethod
    def copy_from(cls, other: Union[JobTemplate, Job]) -> JobTemplate:
        """
        Creates a JobTemplate copy from another JobTemplate or Job instance.
        Copies metadata and converts operations to JobOperation objects.
        """
        new_template = cls(
            id=other.id,
            routing_id=other.routing_id,
            experiment_id=other.experiment_id,
            arrival=other.arrival,
            deadline=other.deadline,
            operations=[]
        )

        # Operationen kopieren
        for op in other.operations:
            new_op = JobOperation(
                job=new_template,
                position_number=op.position_number,
                machine_name=op.machine_name,
                duration=op.duration,
                start=op.start,
                end=op.end
            )
            new_template.operations.append(new_op)

        return new_template


@dataclass
class Operation:
    job_id: str
    position_number: int
    machine_name: str


@dataclass
class JobOperation:
    job: Union[Job, JobTemplate]
    position_number: int
    machine_name: str
    duration: int

    shift_number: Optional[int] = None

    start: Optional[float] = None
    end: Optional[float] = None

    operation: Operation = field(init=False)

    def __repr__(self) -> str:
        attrs = {
            "job_id": self.job.id,
            "position_number": self.position_number,
            "machine_name": self.machine_name,
            "duration": self.duration,
        }
        return "JobOperation(" + ", ".join(f"{key}={value!r}" for key, value in attrs.items()) + ")"

    @property
    def job_id(self) -> str:
        return self.job.id

    @property
    def job_arrival(self) -> int:
        return self.job.arrival

    @property
    def job_earliest_start(self) -> int:
        return self.job.earliest_start

    @property
    def job_deadline(self) -> int:
        return self.job.deadline

    @property
    def routing_id(self) -> str:
        return self.job.routing_id

    @property
    def experiment_id(self) -> Optional[int]:
        return self.job.experiment_id

    def __post_init__(self):
        if self.operation is None:
            self.operation = Operation(
                job_id=self.job_id,
                position_number=self.position_number,
                machine_name=self.machine_name
            )

    def __eq__(self, other):
        if not isinstance(other, JobOperation):
            return NotImplemented
        return self.operation == other.operation

    def __hash__(self):
        return hash(self.operation)