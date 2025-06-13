import pandas as pd
from collections import defaultdict

# FCFS with Arrivals
def schedule_fcfs_with_arrivals(df_jssp: pd.DataFrame, arrival_df: pd.DataFrame) -> pd.DataFrame:
    """
    FCFS-Scheduling mit Job-Ankunftszeiten – optimierte Version.

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - arrival_df: DataFrame mit ['Job','Arrival'].
    """
    # Arrival-Zeiten als Dict
    arrival = arrival_df.set_index('Job')['Arrival'].to_dict()

    # Preprocessing: Operationen als Dict (Job, Operation) → Row
    ops_dict = {(row['Job'], row['Operation']): row for _, row in df_jssp.iterrows()}

    # Status-Tracker
    next_op = {job: 0 for job in df_jssp['Job'].unique()}
    job_ready = arrival.copy()
    machine_ready = defaultdict(float)
    remaining = len(df_jssp)

    schedule = []
    while remaining > 0:
        best = None  # (job, start, dur, machine, op_idx)

        # Suche FCFS-geeignete Operation
        for job, op_idx in next_op.items():
            if (job, op_idx) not in ops_dict:
                continue
            row = ops_dict[(job, op_idx)]
            m = int(row['Machine'].lstrip('M'))  # optional: in ops_dict vorverarbeiten
            dur = row['Processing Time']
            earliest = max(job_ready[job], machine_ready[m])
            if (best is None or
                earliest < best[1] or
                (earliest == best[1] and arrival[job] < arrival[best[0]])):
                best = (job, earliest, dur, m, op_idx)

        job, start, dur, m, op_idx = best
        end = start + dur
        schedule.append({
            'Job': job,
            'Operation': op_idx,
            'Arrival': arrival[job],
            'Machine': f'M{m}',
            'Start': start,
            'Processing Time': dur,
            'End': end
        })
        job_ready[job] = end
        machine_ready[m] = end
        next_op[job] += 1
        remaining -= 1

    df_schedule = pd.DataFrame(schedule)
    return df_schedule.sort_values(['Arrival', 'Start']).reset_index(drop=True)