from sqlalchemy import Column, Integer, String, ForeignKey, PrimaryKeyConstraint, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Instance(Base):
    __tablename__ = "instance"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    routings = relationship("Routing", back_populates="instance", cascade="all, delete-orphan")


class Routing(Base):
    __tablename__ = "routing"

    id = Column(String(255), primary_key=True)
    instance_id = Column(Integer, ForeignKey("instance.id"), nullable=False)
    instance = relationship("Instance", back_populates="routings")
    operations = relationship("Operation", back_populates="routing", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="routing", cascade="all, delete-orphan")


class Operation(Base):
    __tablename__ = "operation"

    routing_id = Column(String(255), ForeignKey("routing.id"), nullable=False)
    number = Column(Integer, nullable=False)  # Reihenfolge innerhalb des Routings
    machine = Column(String(255), nullable=False)
    duration = Column(Integer, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint('routing_id', 'number', name='pk_routing_operation'),
    )

    routing = relationship("Routing", back_populates="operations")


class Job(Base):
    __tablename__ = "job"

    id = Column(String(255), primary_key=True)  # z.B. "J25-0001"
    routing_id = Column(String(255), ForeignKey("routing.id"), nullable=False)
    arrival = Column(Integer, nullable=False)
    ready_time = Column(Integer, nullable=False)
    deadline = Column(Integer, nullable=False)

    routing = relationship("Routing", back_populates="jobs")


# ----------------------------------------------------------------------------------------------------------------------
class Experiment(Base):
    __tablename__ = "experiment"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=True)
    param_settings = Column(JSON, nullable=False)


class Schedule(Base):
    __tablename__ = "schedule"

    experiment_id = Column(Integer, ForeignKey("experiment.id"), nullable=False)
    job_id = Column(String(255), ForeignKey("job.id"), nullable=False)
    operation_id = Column(Integer, nullable=False)     # ← lose Referenz auf operation.number
    day = Column(Integer, nullable=False)

    start = Column(Integer, nullable=False)
    duration = Column(Integer, nullable=False)
    end = Column(Integer, nullable=False)
    machine = Column(String(255), nullable=False)
    log = Column(JSON, nullable=True)

    # Beziehungen
    job = relationship("Job", backref="schedules")
    experiment = relationship("Experiment", backref="schedules")

    __table_args__ = (
        PrimaryKeyConstraint("experiment_id", "job_id", "operation_id", "day", name="pk_schedule"),
    )

    def check_duration(self) -> bool:
        """
        Checks whether the duration is consistent with start and end time.

        :return: True if (start + duration == end), otherwise False
        :rtype: bool
        """
        return self.start + self.duration == self.end
