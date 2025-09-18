from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import numpy as np
from dataclasses import dataclass, field, replace
from sqlalchemy import Column, Integer, String, ForeignKey, ForeignKeyConstraint, Float, Table, Numeric, \
    UniqueConstraint
from sqlalchemy.orm import relationship, backref
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

@mapper_registry.mapped
@dataclass
class Machine:
    __tablename__ = "machine"
    __sa_dataclass_metadata_key__ = "sa"

    id: int = field(init=False, metadata={
        "sa": Column(Integer, primary_key=True, autoincrement=True)
    })

    name: str = field(metadata={
        "sa": Column(String(100), nullable=False, unique=True)
    })

    source_id: int = field(init = False, metadata={
        "sa": Column(Integer, ForeignKey("routing_source.id"), nullable=False)
    })

    # RoutingSource.machines entsteht automatisch durch backref
    source: RoutingSource = field(repr=False, metadata={
        "sa": relationship(
            "RoutingSource",
            lazy="joined",
            backref=backref("machines", cascade="all, delete-orphan")
        )
    })


    def __post_init__(self):
        if self.source_id is None and self.source:
            self.source_id = self.source.id


@mapper_registry.mapped
@dataclass
class MachineInstance:
    __tablename__ = "machine_instance"
    __sa_dataclass_metadata_key__ = "sa"

    id: int = field(init=False, metadata={
        "sa": Column(Integer, primary_key=True, autoincrement=True)
    })

    machine_id: int = field(init =False, metadata={
        "sa": Column(Integer, ForeignKey("machine.id"), nullable=False)
    })

    # Machine.instances entsteht automatisch durch backref
    machine: Machine = field(repr=False, metadata={
        "sa": relationship(
            "Machine",
            backref=backref("instances", cascade="all, delete-orphan")
        )
    })

    max_bottleneck_utilization: Decimal = field(metadata={
        "sa": Column(Numeric(5, 4), nullable=False)
    })

    transition_time: int = field(default=0, metadata={
        "sa": Column(Integer, nullable=False)
    })

    __table_args__ = (
        UniqueConstraint("machine_id", "max_bottleneck_utilization",
                         name="uq_machineinstance_utilization"),
    )

    @property
    def name(self) -> str:
        return self.machine.name

    @property
    def source_id(self) -> int:
        return self.machine.source_id

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

    # backref erzeugt automatisch: RoutingSource.routings
    routing_source: Optional[RoutingSource] = field(
        default=None, repr=False, metadata={
            "sa": relationship(
                "RoutingSource",
                lazy="joined",
                backref=backref("routings", cascade="all, delete-orphan")
            )
        }
    )

    operations: List[RoutingOperation] = field(default_factory=list, repr=False, metadata={"sa": relationship(
                "RoutingOperation", back_populates="routing",
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

    machine_id: int = field(init=False, metadata={
        "sa": Column(Integer, ForeignKey("machine.id"), nullable=False)
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

    machine: Machine = field(
        default=None,
        repr=False,
        metadata={
            "sa": relationship("Machine", lazy="joined")
        }
    )

    @property
    def machine_name(self) -> str:
        return self.machine.name



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
            "due_date": self.due_date,
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

    due_date: Optional[int] = field(default=None, metadata={"sa": Column(Integer, nullable=True)})

    routing: Routing = field(default=None, repr=False, metadata={
        "sa": relationship("Routing", lazy="joined",
                           backref=backref("jobs", cascade="all, delete-orphan"))
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
        "sa": Column(String, ForeignKey("job.id"), primary_key=True, nullable=False)
    })
    experiment_id: int = field(metadata={
        "sa": Column(Integer, ForeignKey("experiment.id"), primary_key=True, nullable=False)
    })

    # Nur lesen, keine PK-Synchronisation:
    job: Job = field(default=None, repr=False, metadata={
        "sa": relationship(
            "Job",
            uselist=False,
            lazy="joined",
            viewonly=True,
            primaryjoin="foreign(SimulationJob.id) == Job.id",
        )
    })
    experiment: Experiment = field(default=None, repr=False, metadata={
        "sa": relationship(
            "Experiment",
            lazy="joined",
            viewonly=True,
            primaryjoin="foreign(SimulationJob.experiment_id) == Experiment.id",
        )
    })

    operations: list[SimulationOperation] = field(default_factory=list, repr=False, metadata={
        "sa": relationship(
            "SimulationOperation",
            back_populates="simulation_job",
            lazy="selectin",
            cascade="all, delete-orphan",
            single_parent=True,   # <-- wichtig für delete-orphan
            order_by="SimulationOperation.position_number",
            primaryjoin=(
                "and_("
                "SimulationJob.id == foreign(SimulationOperation.job_id), "
                "SimulationJob.experiment_id == foreign(SimulationOperation.experiment_id)"
                ")"
            ),
        )
    })

    # Convenience
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
    def due_date(self) -> int:
        return self.job.due_date

    @property
    def max_bottleneck_utilization(self) -> Decimal:
        return self.job.max_bottleneck_utilization


@mapper_registry.mapped
@dataclass
class ScheduleJob:
    __tablename__ = "schedule_job"
    __sa_dataclass_metadata_key__ = "sa"

    # --- Primärschlüssel ---
    id: str = field(metadata={
        "sa": Column(String, ForeignKey("job.id"), primary_key=True, nullable=False)
    })

    experiment_id: int = field(metadata={
        "sa": Column(Integer, ForeignKey("experiment.id"), primary_key=True, nullable=False)
    })

    shift_number: int = field(metadata={
        "sa": Column(Integer, primary_key=True, nullable=False)
    })

    # --- Beziehungen ---
    experiment: Experiment = field(default=None, repr=False, metadata={
        "sa": relationship(
            "Experiment",
            viewonly=True,  # wichtig: verhindert PK-Nullung
            primaryjoin="foreign(ScheduleJob.experiment_id) == Experiment.id"
        )
    })

    operations: List[ScheduleOperation] = field(default_factory=list, repr=False, metadata={
        "sa": relationship(
            "ScheduleOperation",
            back_populates="schedule_job",
            lazy="selectin",
            cascade="all, delete-orphan",
            order_by="ScheduleOperation.position_number",
            primaryjoin=(
                "and_("
                "ScheduleJob.id == foreign(ScheduleOperation.job_id), "
                "ScheduleJob.experiment_id == foreign(ScheduleOperation.experiment_id), "
                "ScheduleJob.shift_number == foreign(ScheduleOperation.shift_number)"
                ")"
            ),
            foreign_keys=(
                "[ScheduleOperation.job_id, "
                "ScheduleOperation.experiment_id, "
                "ScheduleOperation.shift_number]"
            ),
        )
    })

    job: Job = field(default=None, repr=False, metadata={
        "sa": relationship(
            "Job",
            uselist=False,
            lazy="joined",
            viewonly=True,  # auch hier: nur lesen, kein PK schreiben
            primaryjoin="ScheduleJob.id == foreign(Job.id)",
        )
    })

    # --- Convenience Properties ---
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
    def due_date(self) -> int:
        return self.job.due_date

    @property
    def max_bottleneck_utilization(self) -> Decimal:
        return self.job.max_bottleneck_utilization



# Junction Table Experiment-Job (M:N)
#experiment_job = Table(
#    "experiment_job", mapper_registry.metadata,
#    Column("experiment_id", ForeignKey("experiment.id"), primary_key=True),
#    Column("job_id", ForeignKey("job.id"), primary_key=True)
#)


@mapper_registry.mapped
@dataclass
class Experiment:
    __tablename__ = "experiment"
    __sa_dataclass_metadata_key__ = "sa"

    id: int = field(init=False, metadata={
        "sa": Column(Integer, primary_key=True, autoincrement=True)
    })

    source_id: int = field(init=False, metadata={
        "sa": Column(Integer, ForeignKey("routing_source.id"), nullable=False)
    })
    routing_source: RoutingSource = field(init=True, repr=False, metadata={
        "sa": relationship("RoutingSource")
    })

    # Cost function parameters
    absolute_lateness_ratio: Optional[float] = field(metadata={"sa": Column(Float, nullable=True)})

    inner_tardiness_ratio: Optional[float] = field(metadata={"sa": Column(Float, nullable=True)})

    type: Optional[str] = field(default="Optimization", repr=False, metadata={"sa": Column(String(100), nullable=True)})

    # Other parameters
    max_bottleneck_utilization: Decimal = field(default=Decimal("0.5000"), metadata={
        "sa": Column(Numeric(5, 4), nullable=False)
    })

    sim_sigma: float = field(default=0.0, metadata={
        "sa": Column(Float, nullable=False)
    })

    shift_length: int = field(init = False, default=1440, metadata={
        "sa": Column(Integer, nullable=False)
    })

    schedule_jobs: list[ScheduleJob] = field(default_factory=list, repr=False, metadata={
        "sa": relationship(
            "ScheduleJob",
            back_populates="experiment",
            cascade="all, delete-orphan",
            lazy="selectin"
        )
    })

    simulation_jobs: list[SimulationJob] = field(default_factory=list, repr=False, metadata={
        "sa": relationship(
            "SimulationJob",
            back_populates="experiment",
            cascade="all, delete-orphan",
            lazy="selectin"
        )
    })

    def get_solver_weights(self):
        """
        Calculate integer solver weights for tardiness, earliness, and deviation.

        :returns: Tuple ``(w_t, w_e, w_dev)`` with weights for tardiness, earliness and deviation.
        """
        # 1) Split tardiness/earliness ratio into integer weights
        tardiness_frac = Fraction(self.inner_tardiness_ratio).limit_denominator(100)
        tardiness = tardiness_frac.numerator
        earliness = tardiness_frac.denominator - tardiness

        # 2) Split lateness/deviation ratio into integer factors
        lateness_frac = Fraction(self.absolute_lateness_ratio).limit_denominator(100)
        lateness_factor = lateness_frac.numerator
        dev_factor = lateness_frac.denominator - lateness_factor

        # 3) Calculate the total amount of tardiness + earliness
        amount = tardiness + earliness

        # 4) Final weights for tardiness, earliness, and deviation
        w_t = tardiness * lateness_factor
        w_e = earliness * lateness_factor
        w_dev = amount * dev_factor

        return w_t, w_e, w_dev


    def __post_init__(self):
        if not (Decimal("0") <= self.max_bottleneck_utilization <= Decimal("1")):
            raise ValueError("max_bottleneck_utilization must be between 0 and 1 (inclusive).")

        if not (0 <= self.absolute_lateness_ratio <= 1):
            raise ValueError("absolute_lateness_ratio must be between 0 and 1.")

        if not (0 <= self.inner_tardiness_ratio <= 1):
            raise ValueError("inner_tardiness_ratio must be between 0 and 1.")

        if self.source_id is None and self.routing_source:
            self.source_id = self.routing_source.id

        if self.shift_length is None:
            self.shift_length = 1440



@mapper_registry.mapped
@dataclass
class ScheduleOperation:
    __tablename__ = "schedule_operation"
    __sa_dataclass_metadata_key__ = "sa"

    job_id: str = field(metadata={
        "sa": Column(String, primary_key=True, nullable=False)
    })
    experiment_id: int = field(init= True, metadata={
        "sa": Column(Integer, primary_key=True, nullable=False)
    })
    shift_number: int = field(metadata={
        "sa": Column(Integer, primary_key=True, nullable=False)
    })
    position_number: int = field(metadata={
        "sa": Column(Integer, primary_key=True, nullable=False)
    })

    start: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})
    end: int   = field(default=0, metadata={"sa": Column(Integer, nullable=False)})

    # WICHTIG: viewonly, korrekter primaryjoin und echte foreign_keys-Liste
    schedule_job: ScheduleJob = field(default=None, repr=False, metadata={
        "sa": relationship(
            "ScheduleJob",
            back_populates="operations",
            viewonly=True,
            lazy="joined",
            primaryjoin=(
                "and_("
                "foreign(ScheduleOperation.job_id) == ScheduleJob.id, "
                "foreign(ScheduleOperation.experiment_id) == ScheduleJob.experiment_id, "
                "foreign(ScheduleOperation.shift_number) == ScheduleJob.shift_number"
                ")"
            ),
        )
    })

    __table_args__ = (
        ForeignKeyConstraint(
            ["job_id", "experiment_id", "shift_number"],
            ["schedule_job.id", "schedule_job.experiment_id", "schedule_job.shift_number"],
            ondelete=None,

        ),
    )

    # (Deine Properties bleiben so)
    @property
    def _routing_operation(self) -> "RoutingOperation":
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
        "sa": Column(String, primary_key=True, nullable=False)
    })
    experiment_id: int = field(metadata={
        "sa": Column(Integer, primary_key=True, nullable=False)
    })
    position_number: int = field(metadata={
        "sa": Column(Integer, primary_key=True, nullable=False)
    })

    start: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})
    duration: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})
    end: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})

    # Child -> Parent, nur lesen (wir sammeln Ops separat, keine PK-Propagation nötig)
    simulation_job: SimulationJob = field(default=None, repr=False, metadata={
        "sa": relationship(
            "SimulationJob",
            back_populates="operations",
            lazy="joined",
            viewonly=True,
            primaryjoin=(
                "and_("
                "foreign(SimulationOperation.job_id) == SimulationJob.id, "
                "foreign(SimulationOperation.experiment_id) == SimulationJob.experiment_id"
                ")"
            ),
        )
    })

    __table_args__ = (
        ForeignKeyConstraint(
            ["job_id", "experiment_id"],
            ["simulation_job.id", "simulation_job.experiment_id"]
        ),
    )

    @property
    def _routing_operation(self) -> "RoutingOperation":
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
    due_date: Optional[int] = None

    on_arrival: bool = False
    max_bottleneck_utilization: Optional[Decimal] = None
    operations: List[JobOperation] = field(default_factory=list)

    current_operation: Optional[JobOperation] = None
    current_operation_earliest_start: Optional[int] = None

    def __repr__(self) -> str:
        attrs = {
            "id": self.id,
            "routing_id": self.routing_id,
            "arrival": self.arrival,
            "earliest_start": self.arrival,
            "sum_duration":self.sum_duration,
            "max_bottleneck_utilization": self.max_bottleneck_utilization
        }
        return "LiveJob(" + ", ".join(f"{key}={value!r}" for key, value in attrs.items()) + ")"

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

    def get_next_operation(self, this_position_number: int) -> Optional[JobOperation]:
        """
        Gibt die nächste JobOperation (mit größerer position_number) zurück, falls vorhanden.

        :param this_position_number: position_number der aktuellen Operation
        :return: Nächste JobOperation oder None, falls es keine gibt
        """
        next_ops = [
            op for op in self.operations
            if op.position_number > this_position_number
        ]
        if not next_ops:
            return None
        return min(next_ops, key=lambda op: op.position_number)

    def get_first_operation(self) -> Optional[JobOperation]:
        """
        Gibt die erste Operation dieses Jobs zurück (basierend auf kleinster position_number).

        :return: Erste JobOperation oder None, falls keine vorhanden
        """
        if not self.operations:
            return None
        return min(self.operations, key=lambda op: op.position_number)

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
        Total transition time of all operations from given position for this job (inclusive)
        """
        return sum(op.transition_time for op in self.operations if op.position_number >= position)


    def sum_transition_time(self, position: int) -> int:
        """
        Total transition time of all operations from the given position for this job (inclusive)
        """
        return sum(op.transition_time for op in self.operations if op.position_number >= position)


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
            due_date=other.due_date,
            max_bottleneck_utilization=other.max_bottleneck_utilization,
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

    sim_duration: Optional[int] = None

    # --- echte Simulationszeiten an der Maschine ---
    request_time_on_machine: Optional[int] = field(default=None, repr=False)
    granted_time_on_machine: Optional[int] = field(default=None, repr=False)


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
    def job_due_date(self) -> int:
        return self.job.due_date

    @property
    def routing_id(self) -> str:
        return self.job.routing_id


    @property
    def _unique_operation(self) -> Tuple[str, int]:
        return self.job_id, self.position_number

    @property
    def waiting_time_on_machine(self) -> Optional[int]:
        if self.request_time_on_machine is None or self.granted_time_on_machine is None:
            return None
        return self.granted_time_on_machine - self.request_time_on_machine


    def __eq__(self, other):
        if not isinstance(other, JobOperation):
            return NotImplemented
        return self._unique_operation == other._unique_operation

    def __hash__(self):
        return hash(self._unique_operation)