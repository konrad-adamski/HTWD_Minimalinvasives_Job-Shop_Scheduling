from __future__ import annotations

from decimal import Decimal

import numpy as np
from dataclasses import dataclass, field, replace
from sqlalchemy import Column, Integer, String, ForeignKey, ForeignKeyConstraint, Float, Table, Numeric, \
    UniqueConstraint
from sqlalchemy.orm import relationship
from typing import Optional, List, Union, Set, Iterable, Tuple

from src.domain.orm_setup import mapper_registry


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

    def __eq__(self, other):
        if not isinstance(other, Machine):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    id: int = field(init=False, metadata={
        "sa": Column(Integer, primary_key=True, autoincrement=True)
    })

    source_id: Optional[int] = field(init= False, metadata={
        "sa": Column(Integer, ForeignKey("routing_source.id"), nullable=False)
    })

    source: RoutingSource = field(repr=False, metadata={
        "sa": relationship("RoutingSource", lazy="joined")
    })

    name: str = field(metadata={
        "sa": Column(String(100), nullable=False)
    })

    max_bottleneck_utilization: Decimal = field(metadata={
        "sa": Column(Numeric(5, 4), nullable=False)
    })

    transition_time: int = field(default=0, metadata={
        "sa": Column(Integer, nullable=False)
    })

    __table_args__ = (
        UniqueConstraint("name", "max_bottleneck_utilization", name="uq_machine_name_utilization"),
    )

    def __post_init__(self):
        if self.source_id is None and self.source:
            self.source_id = self.source.id



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


@mapper_registry.mapped
@dataclass
class Job:
    __tablename__ = "job"
    __sa_dataclass_metadata_key__ = "sa"
    def __repr__(self) -> str:
        attrs = {
            "id": self.id,
            "routing_id": self.routing_id,
            "arrival": self.arrival,
            "earliest_start": self.earliest_start,
            "deadline": self.deadline,
            "sum_duration": self.sum_duration,
            "max_bottleneck_utilization": self.max_bottleneck_utilization
        }
        return "Job(" + ", ".join(f"{key}={value!r}" for key, value in attrs.items()) + ")"

    def __eq__(self, other):
        if not isinstance(other, Job):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


    id: str = field(metadata={
        "sa": Column(String(255), nullable=False, primary_key=True)
    })

    max_bottleneck_utilization: Optional[Decimal] = field(metadata={
        "sa": Column(Numeric(5, 4), nullable=True)
    })

    routing_id: str = field(init=False, default="", metadata={
        "sa": Column(String(255), ForeignKey("routing.id"), nullable=False)
    })

    arrival: Optional[int] = field(metadata={"sa": Column(Integer, nullable=True)})

    deadline: Optional[int] = field(default=None, metadata={"sa": Column(Integer, nullable=True)})


    routing: Routing = field(default=None, repr=False, metadata={
        "sa": relationship("Routing", back_populates="jobs", lazy="joined")
    })

    # Kein ORM-Relationship mehr zu JobOperation – stattdessen dynamisch generiert
    @property
    def operations(self) -> List[JobOperation]:
        operations: List[JobOperation] = []
        for routing_op in self.routing.operations:
            operations.append(
                JobOperation(
                    job=self,
                    position_number=routing_op.position_number,
                    machine_name=routing_op.machine_name,
                    duration=routing_op.duration
                )
            )
        return operations

    @property
    def earliest_start(self) -> int:
        if self.arrival is None:
            return 0
        return int(np.ceil((self.arrival + 1) / 1440) * 1440)

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
        if self.routing and self.routing_id is None:
            self.routing_id = self.routing.id
        if self.max_bottleneck_utilization and not (Decimal("0") <= self.max_bottleneck_utilization <= Decimal("1")):
            raise ValueError("max_bottleneck_utilization must be between 0 and 1 (inclusive).")



@mapper_registry.mapped
@dataclass
class SimulationJob:
    __tablename__ = "simulation_job"
    __sa_dataclass_metadata_key__ = "sa"

    id: str = field(metadata={
        "sa": Column(String, ForeignKey("job.id"), primary_key=True)
    })

    experiment_id: int = field(init=False, default=None, metadata={
        "sa": Column(Integer, ForeignKey("experiment.id"), primary_key=True)
    })

    job: Job = field(default=None, repr=False, metadata={
        "sa": relationship("Job", lazy="joined")
    })

    experiment: Experiment = field(default=None, repr=False, metadata={
        "sa": relationship("Experiment", lazy="joined")
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
    def max_bottleneck_utilization(self) -> Decimal:
        return self.job.max_bottleneck_utilization



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

    job: Job = field(default=None, repr=False, metadata={
        "sa": relationship("Job", lazy="joined")
    })

    experiment = relationship("Experiment",overlaps="shift")

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
    def max_bottleneck_utilization(self) -> Decimal:
        return self.job.max_bottleneck_utilization


    __table_args__ = (
        ForeignKeyConstraint(
            ["experiment_id", "shift_number"],
            ["shift.experiment_id", "shift.shift_number"]
        ),
    )



# Junction Table Experiment-Job (M:N)
experiment_job = Table(
    "experiment_job", mapper_registry.metadata,
    Column("experiment_id", ForeignKey("experiment.id"), primary_key=True),
    Column("job_id", ForeignKey("job.id"), primary_key=True)
)


@mapper_registry.mapped
@dataclass
class Experiment:
    __tablename__ = "experiment"
    __sa_dataclass_metadata_key__ = "sa"

    id: int = field(init=False, metadata={
        "sa": Column(Integer, primary_key=True, autoincrement=True)
    })

    # Relevant parameters
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


    max_bottleneck_utilization: Decimal = field(default=Decimal("0.5000"), metadata={
        "sa": Column(Numeric(5, 4), nullable=False)
    })

    sim_sigma: float = field(default=0.0, metadata={
        "sa": Column(Float, nullable=False)
    })

    #  general
    total_shift_number: int = field(init=True, default=None, metadata={
        "sa": Column(Integer, nullable=False)
    })

    shift_length: Optional[int] = field(default=1440, metadata={
        "sa": Column(Integer, nullable=False)
    })

    shifts: List[Shift] = field(default_factory=list, repr=False, metadata={"sa": relationship(
        "Shift", back_populates="experiment", cascade="all, delete-orphan", lazy="joined")
    })

    # Jobs
    _jobs: Set[Job] = field(default_factory=set, init= False, repr=False, metadata={
        "sa": relationship(
            "Job",
            secondary=experiment_job,
            collection_class=set,
            lazy="joined"
        )
    })

    @property
    def jobs(self) -> List[Job]:
        return sorted(self._jobs, key=lambda job: job.arrival)

    @property
    def last_shift_start(self) -> int:
        return self.total_shift_number * self.shift_length

    def add_job(self, job: Job) -> None:
        self._jobs.add(job)

    def add_jobs(self, jobs: Iterable[Job]) -> None:
        eligible_jobs = {
            job for job in jobs
            if job.earliest_start <= self.last_shift_start
               and job.max_bottleneck_utilization == self.max_bottleneck_utilization
        }
        self._jobs.update(eligible_jobs)


    def __post_init__(self):
        if not (Decimal("0") <= self.max_bottleneck_utilization <= Decimal("1")):
            raise ValueError("max_bottleneck_utilization must be between 0 and 1 (inclusive).")

        if not (0 <= self.main_pct <= 1):
            raise ValueError("main_pct must be between 0 and 1.")

        if self.shift_length is None:
            self.shift_length = 1440

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

    experiment: Experiment = field(default=None, repr=False, metadata={
        "sa": relationship(Experiment, back_populates="shifts", lazy="joined")
    })


    schedule_jobs: List[ScheduleJob] = field(default_factory=list, repr=False, metadata={
        "sa": relationship(ScheduleJob, overlaps="experiment")
    })

    @property
    def shift_start(self) -> int:
        if self.experiment.shift_length:
            return self.shift_number * self.experiment.shift_length
        else:
            return self.shift_number * 1440

    @property
    def shift_end(self) -> int:
        if self.experiment.shift_length:
            return (self.shift_number + 1) * self.experiment.shift_length
        else:
            return (self.shift_number + 1) * 1440



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
# View/Helper domain (not ORM models): wrap ORM objects for easy access.

@dataclass
class LiveJob:
    id: str
    routing_id: Optional[str] = None
    arrival: Optional[int] = None
    deadline: Optional[int] = None

    on_arrival: bool = False
    max_bottleneck_utilization: Optional[Decimal] = None
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

    @property
    def first_operation_position_number(self) -> Optional[int]:
        """
        Returns the lowest position_number among all operations,
        i.e., the first technological step of the job.
        """
        if not self.operations:
            return None
        return min(op.position_number for op in self.operations)

    def get_previous_operation(self, this_position_number: int) -> Optional[JobOperation]:
        """
        Gibt die vorherige JobOperation (mit kleinerer position_number) zurück, falls vorhanden.

        :param this_position_number: position_number der aktuellen Operation
        :return: Vorherige JobOperation oder None, falls es keine gibt
        """
        previous_ops = [
            op for op in self.operations
            if op.position_number < this_position_number
        ]
        if not previous_ops:
            return None
        return max(previous_ops, key=lambda op: op.position_number)

    def get_last_operation(self) -> Optional[JobOperation]:
        """
        Gibt die letzte Operation dieses Jobs zurück (basierend auf höchster position_number).

        :return: Letzte JobOperation oder None, falls keine vorhanden
        """
        if not self.operations:
            return None
        return max(self.operations, key=lambda op: op.position_number)


    def sum_left_duration(self, position: int) -> int:
        """
        Total duration of all operations from given position for this job (inclusive)
        """
        return sum(op.duration for op in self.operations if op.position_number >= position)


    def sum_left_transition_time(self, position: int) -> int:
        """
        Total duration of all operations after given position for this job (exclusive)
        """
        return sum(op.transition_time for op in self.operations if op.position_number > position)


    @classmethod
    def copy_from(cls, other: Union[LiveJob, Job]) -> LiveJob:
        """
        Creates a JobTemplate copy from another JobTemplate or Job instance.
        Copies metadata and converts operations to JobOperation objects.
        """
        new_template = cls(
            id=other.id,
            routing_id=other.routing_id,
            arrival=other.arrival,
            deadline=other.deadline,
            operations=[]
        )

        # Operationen kopieren
        for operation in other.operations:
            new_template.add_operation_instance(operation)

        return new_template


    def add_operation_instance(
            self, operation: JobOperation, new_start: Optional[float] = None,
            new_duration: Optional[float] = None, new_end: Optional[float] = None) -> None:

        new_op = replace(
            operation,
            job=self,
            start= operation.start if new_start is None else new_start,
            duration=operation.duration if new_duration is None else new_duration,
            end= operation.end if new_end is None else new_end,
        )


        self.operations.append(new_op)


    def set_transition_times(self, machines: List[Machine]) -> None:
        relevant_machines = {
            machine.name: machine.transition_time
            for machine in machines
            if machine.max_bottleneck_utilization == self.max_bottleneck_utilization
        }

        for op in self.operations:
            op.transition_time = relevant_machines.get(op.machine_name, 0)


@dataclass
class JobOperation:
    job: Union[Job, LiveJob]
    position_number: int
    machine_name: str
    duration: int

    transition_time: int = 0

    shift_number: Optional[int] = None

    start: Optional[int] = None
    end: Optional[int] = None


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
    def _unique_operation(self) -> Tuple[str, int]:
        return self.job_id, self.position_number


    def __eq__(self, other):
        if not isinstance(other, JobOperation):
            return NotImplemented
        return self._unique_operation == other._unique_operation

    def __hash__(self):
        return hash(self._unique_operation)