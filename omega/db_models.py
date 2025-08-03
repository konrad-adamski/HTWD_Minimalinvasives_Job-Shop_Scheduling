from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import Column, Integer, String, ForeignKey, ForeignKeyConstraint, Numeric, CheckConstraint, Float, and_
from sqlalchemy.orm import relationship
from typing import Optional, List
from omega.db_setup import mapper_registry


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

    transition_time: int = field(default=0, metadata={
        "sa": Column(Integer, nullable=False)  # z.B. Rüstzeit oder Umlaufzeit
    })

    # Optional: Beziehung zu RoutingOperations
    operations: List[RoutingOperation] = field(default_factory=list, repr=False, metadata={
        "sa": relationship(
            "RoutingOperation",
            back_populates="machine_entity"
        )
    })

    @classmethod
    def from_machines_dataframe(
            cls, df: pd.DataFrame, machine_column: str = "Machine",
            transition_column: str = "Transition Time", default_transition_time: int = 0) -> List[Machine]:
        """
        Create unique Machine objects from a DataFrame.

        :param df: Input DataFrame containing machine information.
        :param machine_column: Column name with machine identifiers.
        :param transition_column: Optional column with transition times.
        :param default_transition_time: Default transition time if column is missing.
        :return: List of Machine instances.
        """
        df_unique = df.drop_duplicates(subset=machine_column).copy()

        if transition_column not in df.columns:
            df_unique[transition_column] = default_transition_time

        machine_entities = []
        for _, row in df_unique.iterrows():
            machine = cls(
                name=row[machine_column],
                transition_time=int(row[transition_column])
            )
            machine_entities.append(machine)

        return machine_entities


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

    operations: List[RoutingOperation] = field(
        default_factory=list,
        repr=False,
        metadata={
            "sa": relationship(
                "RoutingOperation",
                back_populates="routing",
                cascade="all, delete-orphan",
                lazy="joined"
            )
        }
    )

    jobs: List[Job] = field(
        default_factory=list,
        repr=False,
        metadata={
            "sa": relationship(
                "Job",
                back_populates="routing",
                cascade="all, delete-orphan"
            )
        }
    )

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

    @classmethod
    def from_single_routing_dataframe(
            cls, df_routing: pd.DataFrame, routing_column: str = "Routing_ID", operation_column: str = "Operation",
            machine_column: str = "Machine", duration_column: str = "Processing Time",
            source: Optional[RoutingSource] = None) -> Routing:
        """
        Create a single Routing instance from a DataFrame.

        Assumes that the DataFrame contains only one unique routing ID.
        Duplicate operations (based on routing ID and operation number) are removed.

        :param df_routing: The input DataFrame containing routing and operation information.
        :param routing_column: Column name for the routing ID.
        :param operation_column: Column name for the operation step number.
        :param machine_column: Column name for the machine identifier.
        :param duration_column: Column name for the operation duration.
        :param source: Optional RoutingSource object to associate the routing with.
        :return: A Routing object with its associated operations.
        """
        df_clean = df_routing.drop_duplicates(subset=[routing_column, operation_column], keep="first")

        unique_ids = df_clean[routing_column].unique()
        if len(unique_ids) != 1:
            raise ValueError(
                f"Expected exactly one routing ID, but found: {unique_ids}. "
                f"If you have multiple routings, use `from_multiple_routings_dataframe(...)` instead."
            )

        routing_id = str(unique_ids[0])
        new_routing = cls(id=routing_id, routing_source=source, operations=[])

        for _, row in df_clean.iterrows():
            step_nr = int(row[operation_column])
            machine = str(row[machine_column])
            duration = int(row[duration_column])

            new_routing.operations.append(
                RoutingOperation(
                    routing_id=routing_id,
                    position_number=step_nr,
                    machine=machine,
                    duration=duration
                )
            )

        new_routing.operations.sort(key=lambda op: op.position_number)
        return new_routing

    @classmethod
    def from_multiple_routings_dataframe(
            cls, df_routings: pd.DataFrame, routing_column: str = "Routing_ID", operation_column: str = "Operation",
            machine_column: str = "Machine", duration_column: str = "Processing Time",
            source: Optional[RoutingSource] = None) -> List[Routing]:
        """
        Create multiple Routing instances from a DataFrame with multiple Routing_IDs.

        Internally uses `from_single_routing_dataframe()` for each routing group.

        :param df_routings: DataFrame containing multiple routings.
        :param routing_column: Column with routing IDs.
        :param operation_column: Column with operation step numbers.
        :param machine_column: Column with machine identifiers.
        :param duration_column: Column with durations.
        :param source: Optional RoutingSource to associate with all generated Routing objects.
        :return: List of Routing objects.
        """
        new_routings = []

        for routing_id, group in df_routings.groupby(routing_column):
            new_routing = cls.from_single_routing_dataframe(
                df_routing=group,
                routing_column=routing_column,
                operation_column=operation_column,
                machine_column=machine_column,
                duration_column=duration_column,
                source=source
            )
            new_routings.append(new_routing)

        return new_routings


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

    machine: str = field(init=True, metadata={
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

    machine_entity: Optional[Machine] = field(init=False, default=None, repr=False, metadata={
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

    schedule_operations: List[ScheduleOperation] = field(default_factory=list, metadata={
        "sa": relationship("ScheduleOperation", back_populates="job", cascade="all, delete-orphan")
    })

    simulation_operations: List[SimulationOperation] = field(default_factory=list, metadata={
        "sa": relationship("SimulationOperation", back_populates="job", cascade="all, delete-orphan")
    })


    # Kein ORM-Relationship mehr zu JobOperation – stattdessen dynamisch generiert
    @property
    def operations(self) -> List[JobOperation]:
        operations: List[JobOperation] = []
        for routing_op in self.routing.operations:
            operations.append(JobOperation(job=self, routing_operation=routing_op))
        return operations

    @property
    def earliest_start(self) -> int:
        return int(np.ceil(self.arrival + 1 / 1440) * 1440)

    @property
    def max_bottleneck_utilization(self) -> float:
        return float(self.experiment.max_bottleneck_utilization)

    @property
    def sum_duration(self) -> int:
        return self.routing.sum_duration if self.routing else 0

    def __post_init__(self):
        if self.routing:
            self.routing_id = self.routing.id
        if self.experiment:
            self.experiment_id = self.experiment.id


@dataclass
class JobOperation:
    job: Job
    routing_operation: RoutingOperation


    def __repr__(self) -> str:
        attrs = {
            "job_id": self.job.id,
            "routing_id": self.routing_operation.routing_id,
            "position_number": self.position_number,
            "machine": self.machine,
            "duration": self.duration,
        }
        return "JobOperation(" + ", ".join(f"{key}={value!r}" for key, value in attrs.items()) + ")"

    @property
    def position_number(self) -> int:
        return self.routing_operation.position_number

    @property
    def machine(self) -> str:
        return self.routing_operation.machine

    @property
    def duration(self) -> int:
        return self.routing_operation.duration

    @property
    def job_earliest_start(self) -> int:
        return self.job.earliest_start

    @property
    def job_deadline(self) -> int:
        return self.job.deadline


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

    w_first: int = field(default=1, metadata={
        "sa": Column(Integer, nullable=False)
    })

    max_bottleneck_utilization: float = field(default=0.5, metadata={
        "sa": Column(Numeric(5, 4), nullable=False)
    })

    sim_sigma: float = field(default=0.0, metadata={
        "sa": Column(Float, nullable=False)
    })


    jobs: List[Job] = field(default_factory=list, repr=False, metadata={
        "sa": relationship("Job", back_populates="experiment", cascade="all, delete-orphan", lazy="joined")
    })

    schedule_operations: List[ScheduleOperation] = field(
        default_factory=list,
        repr=False,
        metadata={
            "sa": relationship(
                "ScheduleOperation",
                back_populates="experiment",
                cascade="all, delete-orphan",
                lazy="joined"
            )
        }
    )

    simulation_operations: List[SimulationOperation] = field(
        default_factory=list,
        repr=False,
        metadata={
            "sa": relationship(
                "SimulationOperation",
                back_populates="experiment",
                cascade="all, delete-orphan",
                lazy="joined"
            )
        }
    )

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

    experiment: Experiment = field(default=None, repr=True, metadata={
        "sa": relationship("Experiment", backref="shifts", lazy="joined")
    })

    schedule_operations: List[ScheduleOperation] = field(
        default_factory=list,
        repr=False,
        metadata={
            "sa": relationship(
                "ScheduleOperation",
                primaryjoin=(
                    "and_(Shift.experiment_id == ScheduleOperation.experiment_id, "
                    "Shift.shift_number == ScheduleOperation.shift_number)"
                ),
                back_populates="shift",
                cascade="all, delete-orphan",
                lazy="joined",
                overlaps="experiment,schedule_operations"
            )
        }
    )

    @property
    def shift_start(self) -> int:
        return self.shift_number * self.shift_length

    @property
    def shift_end(self) -> int:
        return self.shift_start + self.shift_length


    def __post_init__(self):
        self.shift_length = 1440


@mapper_registry.mapped
@dataclass
class ScheduleOperation:
    __tablename__ = "schedule_operation"
    __sa_dataclass_metadata_key__ = "sa"

    shift_number: int = field(metadata={"sa": Column(Integer, primary_key=True)})

    experiment_id: int = field(metadata={
        "sa": Column(Integer, ForeignKey("experiment.id"), primary_key=True)
    })

    job_id: str = field(metadata={
        "sa": Column(String, ForeignKey("job.id"), primary_key=True)
    })

    position_number: int = field(metadata={"sa": Column(Integer, primary_key=True)})

    start: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})

    end: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})

    experiment: Experiment = field(default=None, repr=False, metadata={
        "sa": relationship(
            "Experiment",
            back_populates="schedule_operations",
            lazy="joined",
            overlaps="shift,schedule_operations"
        )
    })

    job: Job = field(default=None, repr=False, metadata={
        "sa": relationship("Job", back_populates="schedule_operations", lazy="joined")
    })

    shift: Shift = field(default=None, repr=False, metadata={
        "sa": relationship(
            "Shift",
            primaryjoin=(
                "and_(ScheduleOperation.experiment_id == Shift.experiment_id, "
                "ScheduleOperation.shift_number == Shift.shift_number)"
            ),
            back_populates="schedule_operations",
            lazy="joined",
            overlaps="experiment,schedule_operations"
        )
    })

    __table_args__ = (
        ForeignKeyConstraint(
            ["experiment_id", "shift_number"],
            ["shift.experiment_id", "shift.shift_number"]
        ),
    )

    @property
    def _routing(self) -> Routing:
        return self.job.routing

    @property
    def _routing_operation(self) -> RoutingOperation:
        return self._routing.operations[self.position_number]

    @property
    def machine(self) -> str:
        return self._routing_operation.machine

    @property
    def duration(self) -> int:
        return self._routing_operation.duration


@mapper_registry.mapped
@dataclass
class SimulationOperation:
    __tablename__ = "simulation_operation"
    __sa_dataclass_metadata_key__ = "sa"

    experiment_id: int = field(metadata={
        "sa": Column(Integer, ForeignKey("experiment.id"), primary_key=True)
    })

    job_id: str = field(metadata={
        "sa": Column(String, ForeignKey("job.id"), primary_key=True)
    })

    routing_id: str = field(metadata={
        "sa": Column(String, ForeignKey("routing.id"), nullable=False)
    })

    position_number: int = field(metadata={"sa": Column(Integer, primary_key=True)})

    start: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})

    duration: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})

    end: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})


    experiment: Experiment = field(default=None, repr=False, metadata={
        "sa": relationship("Experiment", back_populates="simulation_operations")
    })

    job: Job = field(default=None, repr=False, metadata={
        "sa": relationship("Job", back_populates="simulation_operations", lazy="joined")
    })

    @property
    def _routing(self) -> Routing:
        return self.job.routing

    @property
    def _routing_operation(self) -> RoutingOperation:
        return self._routing.operations[self.position_number]

    @property
    def machine(self) -> str:
        return self._routing_operation.machine

    @property
    def route_duration(self) -> int:
        return self._routing_operation.duration


