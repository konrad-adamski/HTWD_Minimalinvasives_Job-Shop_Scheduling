from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint, ForeignKeyConstraint
from sqlalchemy.orm import registry, relationship
from typing import Optional, List
mapper_registry = registry()


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
class Routing:
    __tablename__ = "routing"
    __sa_dataclass_metadata_key__ = "sa"

    # Technischer Primärschlüssel – wird von der DB generiert
    id: Optional[int] = field(init=False, default=None, metadata={
        "sa": Column(Integer, primary_key=True, autoincrement=True)
    })

    # Domänen-ID
    routing_id: str = field(default="", metadata={
        "sa": Column(String(255), unique=False, nullable=False)
    })

    routing_source_id: Optional[int] = field(default=0, metadata=
    {"sa": Column(Integer, ForeignKey("routing_source.id"), nullable=True)})

    routing_source: Optional[RoutingSource] = field(
        default=None,
        repr=False,
        metadata={
            "sa": relationship("RoutingSource", back_populates="routings")
        }
    )

    # Relationship zu RoutingOperation
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
            "sa": relationship("Job", back_populates="routing")
        }
    )

    __table_args__ = (
        UniqueConstraint("routing_id", "routing_source_id", name="unique_routing_in_source"),
    )

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
        new_routing = cls(routing_id=routing_id, routing_source=source, operations=[])

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

    # Datenbankspalten
    id: Optional[int] = field(init=False, default=None, metadata={
        "sa": Column(Integer, primary_key=True, autoincrement=True)
    })
    routing_id: str = field(default="", metadata={
        "sa": Column(String(255), ForeignKey("routing.id"), nullable=False)
    })
    position_number: int = field(default=0, metadata={
        "sa": Column(Integer, nullable=False)
    })
    machine: str = field(default="", metadata={
        "sa": Column(String(255), nullable=False)
    })
    duration: int = field(default=0, metadata={
        "sa": Column(Integer, nullable=False)
    })

    # Pflichtbeziehung zu Routing – KEIN Optional
    routing: Routing = field(
        default=None,
        repr=False,
        metadata={
            "sa": relationship("Routing", back_populates="operations")
        }
    )

    __table_args__ = (
        UniqueConstraint("routing_id", "position_number", name="unique_routing_position"),
    )

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
        "sa": Column(String, ForeignKey("job.id"), nullable=False)
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

    job: Job = field(
        default=None,
        repr=False,
        metadata={
            "sa": relationship("Job", back_populates="operations")
        }
    )

    @property
    def machine(self) -> str:
        return self.routing_operation.machine

    @property
    def duration(self) -> int:
        return self.routing_operation.duration

    __table_args__ = (
        ForeignKeyConstraint(
            ["routing_id", "position_number"],
            ["routing_operation.routing_id", "routing_operation.position_number"]
        ),
    )


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session


    # Example with multiple Routings
    data = {
        "Routing_ID": ["R1", "R1", "R2"],
        "Operation": [10, 20, 10],
        "Machine": ["M1", "M2", "M3"],
        "Processing Time": [5, 10, 7]
    }
    dframe_routings = pd.DataFrame(data)

    # RoutingSource erzeugen
    routing_source = RoutingSource(name="Mein Test")


    # Routing aus DataFrame erzeugen
    routings = Routing.from_multiple_routings_dataframe(
        dframe_routings,
        source=routing_source
    )

    for routing in routings:
        print(f"Routing-ID: {routing.routing_id} from {routing.routing_source}")
        print(f"Gesamtdauer: {routing.sum_duration} min")

        for op in routing.operations:
            for op in routing.operations:
                print(f"  • Step {op.position_number}: {op.machine}, {op.duration} min")


    # SQLite-Datenbank (lokal in Datei)
    engine = create_engine("sqlite:///routings.db")  # oder sqlite:///:memory: (im RAM)

    # ❌ Alles löschen
    mapper_registry.metadata.drop_all(engine)
    # ✅ Neu erzeugen
    mapper_registry.metadata.create_all(engine)

    # Datenbank-Sitzung starten und Routings speichern
    with Session(engine) as session:
        session.add_all(routings)  # Alle Routings inklusive Operations
        session.commit()

        # Kontrolle: Anzahl gespeicherter Operationen
        num_ops = session.query(RoutingOperation).count()
        print(f"\n{num_ops} RoutingOperations erfolgreich gespeichert.")


