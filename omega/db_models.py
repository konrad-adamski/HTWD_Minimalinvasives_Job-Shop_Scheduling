from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import Column, Integer, String, ForeignKey, ForeignKeyConstraint, Numeric, CheckConstraint, Float
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
            "sa": relationship("RoutingSource", back_populates="routings")
        }
    )

    operations: List[RoutingOperation] = field(
        default_factory=list,
        repr=False,
        metadata={
            "sa": relationship(
                "RoutingOperation",
                back_populates="routing",
                cascade="all, delete-orphan"
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

    id: str = field(default="", metadata={
        "sa": Column(String(255), nullable=False, primary_key=True)
    })

    routing_id: str = field(init= False, default="", metadata={
        "sa": Column(String(255), ForeignKey("routing.id"), nullable=False)
    })

    # Zeitinformationen
    arrival: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})

    deadline: Optional[int] = field(default=None, metadata={"sa": Column(Integer, nullable=False)})

    max_bottleneck_utilization: float = field(default=0.0, metadata={
        "sa": Column(Numeric(5, 4), nullable=False)  # 10 Stellen gesamt, 4 nach dem Komma
    })

    experiment_id: int = field(init=False, default=None, metadata={
        "sa": Column(Integer, ForeignKey("experiment.id"), nullable=False)
    })

    experiment: Experiment = field(default=None, repr=False, metadata={
        "sa": relationship("Experiment", back_populates="jobs")
    })

    # Relations
    routing: Routing = field(default=None, repr=False, metadata={
        "sa": relationship(
            "Routing",
            back_populates="jobs"
        )
    })

    # Beziehung zu JobOperationen
    operations: List[JobOperation] = field(init=False, default_factory=list, repr=False, metadata={
        "sa": relationship(
            "JobOperation",
            back_populates="job",
            cascade="all, delete-orphan"
        )
    })

    @property
    def earliest_start(self) -> int:
        return int(np.ceil(self.arrival + 1 / 1440) * 1440)

    def __post_init__(self):

        self.routing_id = self.routing.id
        self.experiment_id = self.experiment.id

        if not (0 <= self.max_bottleneck_utilization <= 1):
            raise ValueError("max_bottleneck_utilization must be between 0 and 1.")

        for op in self.routing.operations:
            JobOperation(
                routing_operation=op,
                job=self
            )

    # Custom-Property für die Planungslogik
    @property
    def sum_duration(self) -> int:
        return self.routing.sum_duration

    __table_args__ = (
        CheckConstraint("max_bottleneck_utilization >= 0 AND max_bottleneck_utilization <= 1",
                        name="check_utilization_range"),
    )

@mapper_registry.mapped
@dataclass
class JobOperation:
    __tablename__ = "job_operation"
    __sa_dataclass_metadata_key__ = "sa"

    job_id: str = field(init=False, metadata={
        "sa": Column(String, ForeignKey("job.id"), primary_key=True)
    })

    position_number: int = field(init=False, metadata={
        "sa": Column(Integer, primary_key=True)
    })

    routing_id: str = field(init=False, metadata={
        "sa": Column(String, nullable=False)
    })

    # Beziehung zur RoutingOperation
    routing_operation: RoutingOperation = field(metadata={
        "sa": relationship(
            "RoutingOperation",
            primaryjoin=(
                "and_(JobOperation.routing_id == RoutingOperation.routing_id, "
                "JobOperation.position_number == RoutingOperation.position_number)"
            ),
            lazy="joined",
            viewonly=True
        )
    })

    # Beziehung zum übergeordneten Job
    job: Job = field(default=None, repr=False, metadata={
        "sa": relationship("Job", back_populates="operations")
    })

    __table_args__ = (
        ForeignKeyConstraint(
            ["routing_id", "position_number"],
            ["routing_operation.routing_id", "routing_operation.position_number"]
        ),
    )

    def __post_init__(self):
        # IDs aus Beziehungen automatisch setzen
        if self.job:
            self.job_id = self.job.id
        if self.routing_operation:
            self.routing_id = self.routing_operation.routing_id
            self.position_number = self.routing_operation.position_number


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
        return self.job.earliest_start

    @property
    def job_max_bottleneck_utilization(self) -> float:
        return self.job.max_bottleneck_utilization

@mapper_registry.mapped
@dataclass
class Experiment:
    __tablename__ = "experiment"
    __sa_dataclass_metadata_key__ = "sa"

    id: int = field(init=False, metadata={
        "sa": Column(Integer, primary_key=True, autoincrement=True)
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

    def __post_init__(self):
        if not (0 <= self.max_bottleneck_utilization <= 1):
            raise ValueError("max_bottleneck_utilization must be between 0 and 1.")
        if not (0 <= self.main_pct <= 1):
            raise ValueError("main_pct must be between 0 and 1.")

    jobs: List[Job] = field(default_factory=list, repr=False, metadata={
        "sa": relationship("Job", back_populates="experiment", cascade="all, delete-orphan")
    })

    def get_jobs_by_shift_start(self, shift_start: int) -> List[Job]:
        """
        Gibt alle Jobs zurück, deren earliest_start dem gegebenen shift_start entspricht
        und deren max_bottleneck_utilization gleich der des Experiments ist.

        :param shift_start: Startzeit des Shifts (in Minuten)
        :return: gefilterte Liste von Job-Objekten
        """
        jobs: List[Job] = []

        for job in self.jobs:
            if (job.earliest_start == shift_start and
                    float(job.max_bottleneck_utilization) == float(self.max_bottleneck_utilization)):
                jobs.append(job)
        return jobs


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

    shift_length: int = field(default=1440, metadata={
        "sa": Column(Integer, nullable=False)
    })

    experiment: Experiment = field(default=None, repr=True, metadata={
        "sa": relationship("Experiment", backref="shifts")
    })

    @property
    def shift_start(self) -> int:
        return self.shift_number * self.shift_length

    @property
    def shift_end(self) -> int:
        return self.shift_start + self.shift_length

    @property
    def jobs(self) -> List[Job]:
        return self.experiment.get_jobs_by_shift_start(self.shift_start)


@mapper_registry.mapped
@dataclass
class ScheduleJobOperation:
    __tablename__ = "schedule_job_operation"
    __sa_dataclass_metadata_key__ = "sa"

    shift_number: int = field(metadata={"sa": Column(Integer, primary_key=True)})

    experiment_id: int = field(metadata={"sa": Column(Integer, primary_key=True)})

    job_id: str = field(metadata={"sa": Column(String, primary_key=True)})

    position_number: int = field(metadata={"sa": Column(Integer, primary_key=True)})

    routing_id: Optional[str] = field(metadata={"sa": Column(String, default=None)})

    start: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})

    end: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})

    job_operation: JobOperation = field(default=None, repr=False, metadata={
        "sa": relationship(
            "JobOperation",
            primaryjoin=(
                "and_(ScheduleJobOperation.job_id == JobOperation.job_id, "
                "ScheduleJobOperation.position_number == JobOperation.position_number)"
            ),
            lazy="joined"
        )
    })

    shift: Shift = field(default=None, repr=False, metadata={
        "sa": relationship(
            "Shift",
            primaryjoin=(
                "and_(ScheduleJobOperation.experiment_id == Shift.experiment_id, "
                "ScheduleJobOperation.shift_number == Shift.shift_number)"
            ),
            lazy="joined"
        )
    })

    __table_args__ = (
        ForeignKeyConstraint(
            ["job_id", "position_number"],
            ["job_operation.job_id", "job_operation.position_number"]
        ),
        ForeignKeyConstraint(
            ["experiment_id", "shift_number"],
            ["shift.experiment_id", "shift.shift_number"]
        ),
    )

    @property
    def machine(self) -> str:
        return self.job_operation.machine

    @property
    def duration(self) -> int:
        return self.job_operation.duration

@mapper_registry.mapped
@dataclass
class SimulationJobOperation:
    __tablename__ = "simulation_job_operation"
    __sa_dataclass_metadata_key__ = "sa"

    experiment_id: int = field(metadata={
        "sa": Column(Integer, ForeignKey("experiment.id"), primary_key=True)  # ⬅️ HIER!
    })

    job_id: str = field(metadata={"sa": Column(String, primary_key=True)})

    position_number: int = field(metadata={"sa": Column(Integer, primary_key=True)})

    routing_id: Optional[str] = field(metadata={"sa": Column(String, default=None)})

    start: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})

    duration: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})

    end: int = field(default=0, metadata={"sa": Column(Integer, nullable=False)})

    experiment: Experiment = field(default=None, repr=False, metadata={
        "sa": relationship("Experiment", backref="simulation_operations")
    })

    job_operation: JobOperation = field(default=None, repr=False, metadata={
        "sa": relationship(
            "JobOperation",
            primaryjoin=(
                "and_(SimulationJobOperation.job_id == JobOperation.job_id, "
                "SimulationJobOperation.position_number == JobOperation.position_number)"
            ),
            lazy="joined"
        )
    })


    __table_args__ = (
        ForeignKeyConstraint(
            ["job_id", "position_number"],
            ["job_operation.job_id", "job_operation.position_number"]
        ),
    )

    @property
    def machine(self) -> str:
        return self.job_operation.machine

    @property
    def route_duration(self) -> int:
        return self.job_operation.duration


if __name__ == "__main__":

    # RoutingSource erzeugen
    routing_source = RoutingSource(name="Testdatensatz")

    # Example with multiple Routings
    data = {
        "Routing_ID": ["R1", "R1", "R2", "R2"],
        "Operation": [10, 20, 10, 20],
        "Machine": ["M1", "M2", "M3", "M1"],
        "Processing Time": [5, 10, 7, 14]
    }
    dframe_routings = pd.DataFrame(data)

    # Routings aus DataFrame erzeugen
    routings = Routing.from_multiple_routings_dataframe(dframe_routings, source=routing_source)

    for routing in routings:
        print(f"Routing-ID: {routing.id} from {routing.source_name} ({routing.source_id})")
        print(f"Gesamtdauer: {routing.sum_duration} min")

        for op in routing.operations:
            print(f"  • Step {op.position_number}: {op.machine}, {op.duration} min")

    print("\n", "-"*60)

    routing_r1 = next((r for r in routings if r.id == "R1"), None)
    print(f"Routing-ID: {routing_r1.id}")

    # Job Experiment -----------------------------------------------------

    experiment = Experiment()
    print(f"Experiment ID: {experiment.id}, max_utilization: {experiment.max_bottleneck_utilization}")
    job = Job(
        id="J1",
        routing=routing_r1,
        arrival=0,
        deadline=2800,
        experiment=experiment
    )

    jobs = []
    jobs.append(job)

    # 5. Ausgabe
    print(f"\nJob {job.id} (Routing: {job.routing_id}) mit {len(job.operations)} Operationen:")

    for job in jobs:
        for op in job.operations:
            print(f"  – Step {op.position_number}: {op.machine}, {op.duration} min. "
                  f"Job earliest_start: {op.job_earliest_start}")

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    
    # SQLite-Datenbank (lokal in Datei)
    engine = create_engine("sqlite:///factory.db")  # oder sqlite:///:memory: (im RAM)

    # ❌ Alles löschen
    mapper_registry.metadata.drop_all(engine)
    # ✅ Neu erzeugen
    mapper_registry.metadata.create_all(engine)

    # 3. Session starten
    with Session(engine) as session:

        # Optional: Machines hinzufügen, falls benötigt (hier ohne transition time)
        machines = Machine.from_machines_dataframe(dframe_routings)
        session.add_all(machines)

        # RoutingSource + Routings (mit Operations) speichern
        session.add(routing_source)     # enthält auch die Routings per backref
        #session.add_all(routings)       # redundant, aber sicher

        # Job + JobOperation speichern
        #session.add(job)  # job enthält job.operations über backref

        # Abschicken
        session.commit()


