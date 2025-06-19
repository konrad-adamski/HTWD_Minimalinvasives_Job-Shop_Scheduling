import pandas as pd
from collections import defaultdict


def schedule(df_jssp: pd.DataFrame, job_column: str = "Job") -> pd.DataFrame:
    """
    Führt FCFS-Scheduling (First Come, First Served) ohne Ankunftszeiten durch.

    Parameter:
    - df_jssp: DataFrame mit Spalten [job_column, 'Operation', 'Machine', 'Processing Time']
    - job_column: Name der Spalte, die den Produktionsauftrag (Job) eindeutig identifiziert

    Rückgabe:
    - df_schedule: DataFrame mit Spalten [job_column, 'Operation', 'Machine', 'Start', 'Processing Time', 'End']
    """
    next_op = {job: 0 for job in df_jssp[job_column].unique()}
    job_ready = {job: 0.0 for job in df_jssp[job_column].unique()}
    machine_ready = defaultdict(float)
    remaining = len(df_jssp)

    schedule = []

    while remaining > 0:
        best = None

        for job, op_idx in next_op.items():
            if op_idx >= (df_jssp[job_column] == job).sum():
                continue
            row = df_jssp[(df_jssp[job_column] == job) & (df_jssp['Operation'] == op_idx)].iloc[0]
            machine = row['Machine']
            dur = row['Processing Time']
            earliest = max(job_ready[job], machine_ready[machine])
            if (best is None or
                earliest < best[1] or
                (earliest == best[1] and job < best[0])):  # alphabetisch als Tiebreaker
                best = (job, earliest, dur, machine, op_idx)

        job, start, dur, machine, op_idx = best
        end = start + dur
        schedule.append({
            job_column: job,
            'Operation': op_idx,
            'Machine': machine,
            'Start': start,
            'Processing Time': dur,
            'End': end
        })

        job_ready[job] = end
        machine_ready[machine] = end
        next_op[job] += 1
        remaining -= 1

    df_schedule = pd.DataFrame(schedule).sort_values([job_column, 'Start']).reset_index(drop=True)
    makespan = df_schedule['End'].max()

    print("\nSchedule-Informationen:")
    print(f"  Makespan: {makespan}")
    return df_schedule


def schedule_with_arrivals(df_jssp: pd.DataFrame, arrival_df: pd.DataFrame,
                                job_column: str = 'Job') -> pd.DataFrame:
    """
    Führt FCFS-Scheduling durch, wobei Job-Ankunftszeiten berücksichtigt werden.

    Parameter:
    - df_jssp: DataFrame mit Spalten [job_column, 'Operation', 'Machine', 'Processing Time']
    - arrival_df: DataFrame mit Spalten [job_column, 'Arrival']
    - job_column: Name der Spalte, die die Jobs eindeutig identifiziert (default: 'Job')

    Rückgabe:
    - DataFrame mit geplanten Operationen und Zeitpunkten
    """
    # Arrival-Zeiten als Dictionary
    arrival = arrival_df.set_index(job_column)['Arrival'].to_dict()

    # Operationen vorbereiten: (Job, Operation) → Zeile
    ops_dict = {(row[job_column], row['Operation']): row for _, row in df_jssp.iterrows()}

    # Initialisierungen
    next_op = {job: 0 for job in df_jssp[job_column].unique()}
    job_ready = arrival.copy()
    machine_ready = defaultdict(float)
    remaining = len(df_jssp)

    schedule = []

    while remaining > 0:
        best = None  # (job, start, dur, machine, op_idx)

        for job, op_idx in next_op.items():
            if (job, op_idx) not in ops_dict:
                continue
            row = ops_dict[(job, op_idx)]
            machine = row['Machine']
            dur = row['Processing Time']
            earliest = max(job_ready[job], machine_ready[machine])

            if (best is None or
                earliest < best[1] or
                (earliest == best[1] and arrival[job] < arrival[best[0]])):
                best = (job, earliest, dur, machine, op_idx)

        job, start, dur, machine, op_idx = best
        end = start + dur
        schedule.append({
            job_column: job,
            'Operation': op_idx,
            'Arrival': arrival[job],
            'Machine': machine,
            'Start': start,
            'Processing Time': dur,
            'End': end
        })

        job_ready[job] = end
        machine_ready[machine] = end
        next_op[job] += 1
        remaining -= 1

    df_schedule = pd.DataFrame(schedule).sort_values([job_column, 'Start']).reset_index(drop=True)
    makespan = df_schedule['End'].max()

    print("\nSchedule-Informationen:")
    print(f"  Makespan: {makespan}")

    return df_schedule