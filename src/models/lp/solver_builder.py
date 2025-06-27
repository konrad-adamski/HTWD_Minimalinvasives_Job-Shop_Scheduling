# Gemeinsame Hilfsfunktionen
import pandas as pd
import pulp
import math
import time
from typing import Tuple, Dict, List

def prepare_jssp_inputs(df_jssp: pd.DataFrame, df_times: pd.DataFrame, job_column: str, earliest_start_column: str, sort_ascending: bool
) -> Tuple[List, List, set, Dict, Dict]:
    df_times = df_times.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    earliest_start = df_times.set_index(job_column)[earliest_start_column].to_dict()
    deadline = df_times.set_index(job_column)["Deadline"].to_dict()
    jobs = df_times[job_column].tolist()

    ops_grouped = df_jssp.sort_values([job_column, "Operation"]).groupby(job_column)
    all_ops, machines = [], set()
    for job in jobs:
        seq = []
        for _, row in ops_grouped.get_group(job).iterrows():
            op_id = row["Operation"]
            m = str(row["Machine"])
            d = float(row["Processing Time"])
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    return jobs, all_ops, machines, earliest_start, deadline


def build_jssp_variables(jobs: List, all_ops: List, earliest_start: Dict, var_cat: str, **extra_vars):
    n = len(jobs)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=earliest_start[jobs[j]], cat=var_cat)
        for j in range(n) for o in range(len(all_ops[j]))
    }

    ends = {
        j: pulp.LpVariable(f"end_{j}", lowBound=earliest_start[jobs[j]], cat=var_cat)
        for j in range(n)
    }

    extras = [
        {
            j: pulp.LpVariable(f"{name}_{j}", **props)
            for j in range(n)
        }
        for name, props in extra_vars.items()
    ]

    return starts, ends, *extras


def add_machine_constraints(prob, all_ops, starts, machines, epsilon, bigM):
    for m in machines:
        ops_on_m = [
            (j, o, seq[o][2])
            for j, seq in enumerate(all_ops)
            for o in range(len(seq))
            if seq[o][1] == m
        ]
        for i in range(len(ops_on_m)):
            j1, o1, d1 = ops_on_m[i]
            for j2, o2, d2 in ops_on_m[i + 1:]:
                if j1 == j2:
                    continue
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM * (1 - y)
                prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM * y

def define_technological_constraints(prob, jobs, all_ops, starts, ends, targets, deadline, mode: str | None = "tardiness"):
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev
        d_last = seq[-1][2]
        prob += ends[j] == starts[(j, len(seq) - 1)] + d_last

        # Zielgrößen-Constraints 
        if mode is not None:
            if mode == "tardiness":
                prob += targets[j] >= ends[j] - deadline[job]
            elif mode == "absolute_lateness":
                lateness = ends[j] - deadline[job]
                prob += targets[j] >= lateness
                prob += targets[j] >= -lateness
            else:
                raise ValueError(f"Unbekannter mode: {mode}")


def get_solver_instance(solver: str, time_limit: int, solver_args: dict):
    solver_args.setdefault("msg", True)
    solver_args.setdefault("timeLimit", time_limit)
    solver = solver.upper()
    if solver == "HIGHS":
        return pulp.HiGHS_CMD(**solver_args)
    elif solver == "CBC":
        return pulp.PULP_CBC_CMD(**solver_args)
    else:
        raise ValueError("Solver must be 'CBC' or 'HiGHS'")


#  Fixed Ops ------------------------------------------------------------------------------------------------
def build_jssp_variables_with_fixed_ops(jobs: List, all_ops: List, earliest_start: Dict,
                                        last_executed_end: Dict, reschedule_start: float,
                                        var_cat: str, **extra_vars):
    """
    Erstellt Start- und Endzeitvariablen mit angepasstem LowBound unter Berücksichtigung
    der letzten ausgeführten Operation je Job (reschedule_start).
    """
    n = len(jobs)

    starts = {
        (j, o): pulp.LpVariable(
            f"start_{j}_{o}",
            lowBound=max(earliest_start[jobs[j]], last_executed_end.get(jobs[j], reschedule_start)),
            cat=var_cat
        )
        for j in range(n)
        for o in range(len(all_ops[j]))
    }

    ends = {
        j: pulp.LpVariable(
            f"end_{j}",
            lowBound=max(earliest_start[jobs[j]], last_executed_end.get(jobs[j], reschedule_start)),
            cat=var_cat
        )
        for j in range(n)
    }

    extras = [
        {
            j: pulp.LpVariable(f"{name}_{j}", **props)
            for j in range(n)
        }
        for name, props in extra_vars.items()
    ]

    return starts, ends, *extras

def add_machine_constraints_with_fixed_ops(prob, all_ops, starts, machines, epsilon, bigM,
                                           fixed_ops: Dict[str, List[Tuple[float, float, str]]]):
    """
    Fügt Maschinenkonflikte zwischen geplanten und fixierten Operationen hinzu.
    """
    for m in machines:
        ops_on_m = [
            (j, o, seq[o][2])
            for j, seq in enumerate(all_ops)
            for o in range(len(seq))
            if seq[o][1] == m
        ]

        # Konflikte zwischen geplanten Operationen
        for i in range(len(ops_on_m)):
            j1, o1, d1 = ops_on_m[i]
            for j2, o2, d2 in ops_on_m[i + 1:]:
                if j1 == j2:
                    continue
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM * (1 - y)
                prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM * y

        # Konflikte mit bereits fixierten Operationen
        for j1, o1, d1 in ops_on_m:
            for fixed_start, fixed_end, fixed_job in fixed_ops.get(m, []):
                y_fix = pulp.LpVariable(f"y_fix_{j1}_{o1}_{fixed_job}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= fixed_start + bigM * (1 - y_fix)
                prob += fixed_end + epsilon <= starts[(j1, o1)] + bigM * y_fix

def define_technological_constraints_with_fixed_ops(prob, jobs, all_ops, starts, ends,
                                                    targets, deadline,
                                                    last_executed_end: Dict[str, float],
                                                    earliest_start: Dict[str, float],
                                                    reschedule_start: float,
                                                    mode: str = "tardiness"):
    """
    Fügt technologische Constraints und Tardiness- bzw. Lateness-Zielgrößen für Jobs mit
    bereits teilweise ausgeführten Operationen hinzu.

    - Startzeit der ersten planbaren Operation ≥ max(Ready Time, Ende letzter Ausführung)
    - `mode`: "tardiness" oder "absolute_lateness"
    """
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        earliest = max(earliest_start[job], last_executed_end.get(job, reschedule_start))
        prob += starts[(j, 0)] >= earliest

        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev

        d_last = seq[-1][2]
        prob += ends[j] == starts[(j, len(seq) - 1)] + d_last

        if mode == "tardiness":
            prob += targets[j] >= ends[j] - deadline[job]
        elif mode == "absolute_lateness":
            lateness = ends[j] - deadline[job]
            prob += targets[j] >= lateness
            prob += targets[j] >= -lateness
        else:
            raise ValueError(f"Unbekannter mode: {mode}")


# ------------------------------------------------------------------------------------------------------------------
        

def get_records_df(df_jssp: pd.DataFrame, df_times: pd.DataFrame, jobs_list: list[str], all_ops: list[list[tuple]], 
                   starts: dict, job_column: str = "Job") -> pd.DataFrame:
    """
    Baut ein robustes Ergebnis-DataFrame direkt aus dem Solver-Modell (jobs_list, all_ops, starts).

    Parameter:
    - df_jssp: Ursprüngliche Operationsdaten (inkl. Maschinen, Processing Times etc.)
    - jobs_list: Reihenfolge der Jobs wie im Modell
    - all_ops: Liste der Operationen pro Job, wie im Modell genutzt
    - starts: Dict mit pulp-Variablen (j, o)
    - df_times: Enthält Zusatzinformationen wie Arrival, Deadline etc.
    - job_column: Spaltenname für die Job-ID
    """
    # 1. Ergebnisse aus Modell extrahieren
    records = []
    for j, job in enumerate(jobs_list):
        for o, (op_id, machine, proc_time) in enumerate(all_ops[j]):
            start = round(starts[(j, o)].varValue, 2)
            end = round(start + proc_time, 2)
            records.append({
                job_column: job,
                "Operation": op_id,
                "Start": start,
                "End": end,
            })

    df_result = pd.DataFrame.from_records(records).sort_values(["Start", job_column, "Operation"]).reset_index(drop=True)

    # 2. Zusatzinformationen konfliktfrei vorbereiten
    shared_cols = set(df_jssp.columns) & set(df_times.columns)
    shared_cols_to_drop = [col for col in shared_cols if col != job_column]
    df_times_filtered = df_times.drop(columns=shared_cols_to_drop, errors="ignore")

    # 3. Merge aus Originaldaten und Zeitinformationen
    df_info = df_jssp.merge(df_times_filtered, on=job_column, how="left")

    # 4. Ergebnis mit den berechneten Start-/Endzeiten zusammenführen
    df_result = df_result.merge(df_info, on=[job_column, "Operation"], how="inner")

    return df_result


def get_schedule_df(jobs, all_ops, starts, df_jssp: pd.DataFrame, df_times: pd.DataFrame, job_column="Job"):
    
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            ed = st + d
            record = {
                job_column: job,
                "Operation": op_id,
                "Machine": m,
                "Start": round(st, 2),
                 "End": round(ed, 2)
            }
            records.append(record)
    df_records = pd.DataFrame.from_records(records)

    # 2a. Kollisionen 
    shared_cols = set(df_jssp.columns) & set(df_times.columns)
    jssp_shared_cols_to_drop = [col for col in shared_cols if col != job_column]
    
    # 2b. Legacy Einträge entfernen
    jssp_legacy_cols = [col for col in ["Start", "End",  "Deadline", "Tardiness", "Lateness","Absolute Lateness", "Flow Time"] if col in df_jssp.columns]
    times_legacy_cols = ["End"] if "End" in df_times.columns else []

    jssp_cols_to_remove = jssp_shared_cols_to_drop + jssp_legacy_cols
    df_jssp_filtered = df_jssp.drop(columns=jssp_cols_to_remove, errors="ignore")
    
    df_times_filtered = df_times.drop(columns=times_legacy_cols, errors="ignore")


    
    # 2c. Merge aus Originaldaten und Zeitinformationen
    df_info = df_jssp_filtered.merge(df_times_filtered, on=job_column, how="left")

    # 3. Ergebnis
    df_schedule = df_info.merge(df_records, on=[job_column, "Operation", "Machine"], how="left")
    
    return df_schedule
    



                