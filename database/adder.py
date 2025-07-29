from typing import List
import pandas as pd
from sqlalchemy import select
from database.db_models import Instance, Routing, Job, Schedule, Experiment, RoutingOperation
from database.db_setup import SessionLocal


def add_routings_and_operations_from_dframe(
        df_routing: pd.DataFrame, instance_name: str, routing_column: str = "Routing_ID",
        operation_column: str = "Operation", machine_column: str = "Machine",
        duration_column: str = "Processing Time") -> List[str]:
    """
    Add routings and their operations from a DataFrame for a given instance.
    Existing routings (by ID) will be skipped.

    :param df_routing: DataFrame with routing and operation data.
    :param instance_name: Target instance name (created if not present).
    :param routing_column: Column name for routing IDs.
    :param operation_column: Column name for operation numbers.
    :param machine_column: Column name for machine identifiers.
    :param duration_column: Column name for processing durations.
    :return: List of skipped routing IDs (because they already existed).
    """
    skipped_routings = []

    with SessionLocal() as session:
        # 1. Ensure instance exists
        instance = session.query(Instance).filter_by(name=instance_name).first()
        if instance is None:
            instance = Instance(name=instance_name)
            session.add(instance)
            session.commit()

        # 2. Process each routing group
        for routing_id, df_group in df_routing.groupby(routing_column):
            routing_id_str = str(routing_id).strip()

            # Skip existing routings (any instance)
            if session.query(Routing).filter_by(id=routing_id_str).first():
                skipped_routings.append(routing_id_str)
                continue

            routing = Routing(id=routing_id_str, instance_id=instance.id)
            session.add(routing)

            for _, row in df_group.iterrows():
                op = RoutingOperation(
                    routing_id=routing_id_str,
                    number=int(row[operation_column]),  # jetzt verpflichtend für PrimaryKey
                    machine=str(row[machine_column]).strip(),
                    duration=int(row[duration_column])
                )
                session.add(op)

        session.commit()

    print(f"✅ {df_routing[routing_column].nunique() - len(skipped_routings)} routings added.")
    if skipped_routings:
        print(f"⚠️  {len(skipped_routings)} routings skipped (already existed): {skipped_routings[:5]} ...")

    return skipped_routings



def add_jobs_from_dataframe(
        df_jobs: pd.DataFrame, job_column: str = "Job",
        routing_column: str = "Routing_ID", arrival_column: str = "Arrival",
        ready_column: str = "Ready Time", deadline_column: str = "Deadline") -> List[str]:
    """
    Add jobs to the database from a DataFrame. Jobs referencing non-existing routing IDs are skipped.

    :param df_jobs: DataFrame containing job data.
    :param job_column: Name of the column containing the job ID.
    :param routing_column: Name of the column containing the routing ID.
    :param arrival_column: Name of the column for job arrival time.
    :param ready_column: Name of the column for job ready time.
    :param deadline_column: Name of the column for job deadline.
    :return: List of skipped job IDs due to missing routing references.
    """
    skipped_jobs = []

    with SessionLocal() as session:
        for _, row in df_jobs.iterrows():
            job_id = str(row[job_column])
            try:
                routing_id = str(int(row[routing_column]))
            except (ValueError, TypeError):
                skipped_jobs.append(job_id)
                continue

            routing_exists = session.execute(
                select(Routing.id).where(Routing.id == routing_id)
            ).scalar() is not None

            if not routing_exists:
                skipped_jobs.append(job_id)
                continue

            job = Job(
                id=job_id,
                routing_id=routing_id,
                arrival=int(row[arrival_column]),
                ready_time=int(row[ready_column]),
                deadline=int(row[deadline_column])
            )
            session.add(job)

        session.commit()

    print(f"✅ {len(df_jobs) - len(skipped_jobs)} jobs added.")
    if skipped_jobs:
        print(f"⚠️  {len(skipped_jobs)} jobs skipped (routing not found): {skipped_jobs[:5]} ...")

    return skipped_jobs

def add_schedule_entries_from_dataframe(
        df: pd.DataFrame, experiment_id: int, day: int, log: dict | list | None,
        job_column: str = "Job", operation_column: str = "Operation", machine_column: str = "Machine",
        start_column: str = "Start", duration_column: str = "Processing Time", end_column: str = "End") -> None:
    """
    Adds schedule entries from a DataFrame to the database for a given experiment and day.

    If entries for (experiment_id, day) already exist, or if the experiment doesn't exist,
    no changes are made and a message is printed instead.

    :param df: DataFrame with schedule data
    :param experiment_id: ID of the experiment
    :param day: Day to assign the entries to
    :param log: Optional log data to store with each entry (same for all)
    :param job_column: Column name for Job ID
    :param operation_column: Column name for Operation number
    :param machine_column: Column name for Machine
    :param start_column: Column name for Start time
    :param duration_column: Column name for Duration
    :param end_column: Column name for End time
    """
    session = SessionLocal()
    try:
        # 1. Check if experiment exists
        experiment = session.query(Experiment).get(experiment_id)
        if experiment is None:
            print(f"❌ Experiment {experiment_id} not found. Nothing written.")
            return

        # 2. Check if schedule already exists
        existing = session.query(Schedule).filter_by(experiment_id=experiment_id, day=day).first()
        if existing:
            print(f"⚠️ Schedule already exists for experiment {experiment_id}, day {day}. Nothing written.")
            return

        # 3. Build schedule entries
        schedule_entries: List[Schedule] = []

        for _, row in df.iterrows():
            entry = Schedule(
                experiment_id=experiment_id,
                job_id=str(row[job_column]),
                operation_id=int(row[operation_column]),
                day=day,
                machine=str(row[machine_column]),
                start=int(row[start_column]),
                duration=int(row[duration_column]),
                end=int(row[end_column]),
                log=log
            )
            schedule_entries.append(entry)

        # 4. Insert
        session.add_all(schedule_entries)
        session.commit()
        print(f"✅ {len(schedule_entries)} schedule entries added for experiment {experiment_id}, day {day}")

    except Exception as e:
        session.rollback()
        print(f"❌ Error while adding schedule entries: {e}")

    finally:
        session.close()