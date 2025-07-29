from typing import List, Optional

import pandas as pd
from sqlalchemy import select
from database.db_models import Job, Routing, Instance, Schedule
from database.db_setup import SessionLocal


def get_jobs_by_instance(
        instance_name: str, job_column: str = "Job", routing_column: str = "Routing_ID",
        arrival_column: str = "Arrival", ready_column: str = "Ready Time",
        deadline_column: str = "Deadline") -> pd.DataFrame:
    """
    Retrieve jobs for a given instance and return them as a DataFrame with configurable column names.

    :param instance_name: Name of the instance (e.g. "Fisher and Thompson 10x10").
    :param job_column: Column name for the job ID.
    :param routing_column: Column name for the routing ID.
    :param arrival_column: Column name for the arrival time.
    :param ready_column: Column name for the ready time.
    :param deadline_column: Column name for the deadline.
    :return: DataFrame containing the selected job data.
    """
    with SessionLocal() as session:
        jobs = session.execute(
            select(Job)
            .join(Routing, Job.routing_id == Routing.id)
            .join(Instance, Routing.instance_id == Instance.id)
            .where(Instance.name == instance_name)
        ).scalars().all()

    if not jobs:
        print(f"⚠️  No jobs found for instance '{instance_name}'.")
        return pd.DataFrame()

    df = pd.DataFrame([{
        job_column: job.id,
        routing_column: job.routing_id,
        arrival_column: job.arrival,
        ready_column: job.ready_time,
        deadline_column: job.deadline
    } for job in jobs])

    print(f"✅ {len(df)} jobs found for instance '{instance_name}'.")
    return df


def get_jssp_by_job_ids(
    job_ids: List[str],
    job_column: str = "Job",
    routing_column: str = "Routing_ID",
    operation_column: str = "Operation",
    machine_column: str = "Machine",
    duration_column: str = "Processing Time"
) -> pd.DataFrame:
    """
    Retrieve JSSP operation data for a list of job IDs and return as a DataFrame,
    including operation index, machine, and duration.

    :param job_ids: List of job IDs to include.
    :param job_column: Column name for job ID.
    :param routing_column: Column name for routing ID.
    :param operation_column: Column name for operation index.
    :param machine_column: Column name for machine.
    :param duration_column: Column name for processing duration.
    :return: DataFrame with job-wise operation data.
    """
    with SessionLocal() as session:
        jobs = session.execute(
            select(Job).where(Job.id.in_(job_ids))
        ).scalars().all()

        if not jobs:
            print("⚠️ No matching jobs found.")
            return pd.DataFrame()

        records = []
        for job in jobs:
            routing = session.query(Routing).filter_by(id=job.routing_id).first()
            if not routing or not routing.operations:
                continue

            sorted_ops = sorted(routing.operations, key=lambda routing_op: routing_op.number)
            for op in sorted_ops:
                records.append({
                    job_column: job.id,
                    routing_column: routing.id,
                    operation_column: op.number,
                    machine_column: op.machine,
                    duration_column: op.duration
                })

        df = pd.DataFrame(records)
        print(f"✅ Retrieved JSSP data for {len(jobs)} jobs with {len(df)} operations.")
        return df

def get_schedule_dataframe(
        experiment_id: int, day: int, job_column: str = "Job", operation_column: str = "Operation",
        machine_column: str = "Machine", start_column: str = "Start", duration_column: str = "Processing Time",
        end_column: str = "End") -> Optional[pd.DataFrame]:
    """
    Returns a schedule for a given experiment and day as a pandas DataFrame
    with custom column names.

    :param experiment_id: ID of the experiment
    :param day: Day to filter
    :param job_column: Output column name for Job ID
    :param operation_column: Output column name for Operation number
    :param machine_column: Output column name for machine
    :param start_column: Output column name for start time
    :param duration_column: Output column name for duration
    :param end_column: Output column name for end time
    :return: DataFrame with renamed schedule fields, or None if no entries found
    """
    session = SessionLocal()

    try:
        results = (
            session.query(Schedule)
            .filter_by(experiment_id=experiment_id, day=day)
            .order_by(Schedule.job_id, Schedule.operation_id)
            .all()
        )

        data = [{
            job_column: schedule_entry.job_id,
            operation_column: schedule_entry.operation_id,
            machine_column: schedule_entry.machine,
            start_column: schedule_entry.start,
            duration_column: schedule_entry.duration,
            end_column: schedule_entry.end,
            } for schedule_entry in results]

        return pd.DataFrame(
            data, columns=[job_column, operation_column, machine_column, start_column, duration_column, end_column]
        )
    except Exception as e:
        print(f"Error in get_schedule_dataframe: {e}")
        return None

    finally:
        session.close()