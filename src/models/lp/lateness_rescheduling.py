from src.models.lp.solver_builder import *

import math
import pulp
import pandas as pd
import time


# Lateness Rescheduling----------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------

# Min. Summe abs. Lateness -----------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------

# mit Deviation Penalty (& fixierte Operation, die hineinlaufen)
def solve_jssp_sum_with_devpen(df_jssp: pd.DataFrame, df_times: pd.DataFrame, df_executed: pd.DataFrame,
                                                 df_original_plan: pd.DataFrame, r: float = 0.5, reschedule_start: float = 1440.0,
                                                 job_column: str = "Job", solver: str = "HiGHS", epsilon: float = 0.0,
                                                 var_cat: str = "Continuous", time_limit: int | None = 10800, sort_ascending: bool = False,
                                                 **solver_args) -> pd.DataFrame:
    """
    Minimiert: Z(σ) = r * sum_j [|C_j - d_j|] + (1 - r) * Summe aller Startzeitabweichungen zum Originalplan
    Berücksichtigt technologische Reihenfolge, Maschinenkonflikte, Fixierungen und Planabweichung.
    """
    start_time = time.time()

    # 1. Vorverarbeitung
    df_times = df_times.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_times.set_index(job_column)["Arrival"].to_dict()
    deadline = df_times.set_index(job_column)["Deadline"].to_dict()
    jobs = df_times[job_column].tolist()

    original_start = {
        (row[job_column], row["Operation"]): row["Start"]
        for _, row in df_original_plan.iterrows()
    }

    # 2. BigM bestimmen
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_deadline = max(deadline.values())
    num_machines = df_jssp["Machine"].nunique()
    bigM = math.ceil((max_deadline - min_arrival + sum_proc_time / math.sqrt(num_machines)) / 1000) * 1000
    print(f"BigM: {bigM}")

    # 3. Operationen und Maschinen
    ops_grouped = df_jssp.sort_values([job_column, "Operation"]).groupby(job_column)
    all_ops, machines = [], set()
    for job in jobs:
        seq = []
        for _, row in ops_grouped.get_group(job).iterrows():
            op_id, m, d = row["Operation"], str(row["Machine"]), float(row["Processing Time"])
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    n = len(jobs)

    # 4. Fixierte Operationen vorbereiten
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    fixed_ops = {
        m: list(g[["Start", "End", job_column]].itertuples(index=False, name=None))
        for m, g in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby(job_column)["End"].max().to_dict()

    # 5. LP-Modell
    prob = pulp.LpProblem("JSSP_AbsLateness_Deviation", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]], cat=var_cat)
        for j in range(n)
        for o in range(len(all_ops[j]))
    }
    ends = {
        j: pulp.LpVariable(f"end_{j}", lowBound=arrival[jobs[j]], cat=var_cat)
        for j in range(n)
    }
    abs_lateness = {
        j: pulp.LpVariable(f"abs_lateness_{j}", lowBound=0, cat=var_cat)
        for j in range(n)
    }
    deviation_vars = {}

    # 6. Constraints & Abweichung zum Originalplan
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        prob += ends[j] == starts[(j, len(seq)-1)] + seq[-1][2]
        lateness = ends[j] - deadline[job]
        prob += abs_lateness[j] >= lateness
        prob += abs_lateness[j] >= -lateness

        for o, (op_id, _, _) in enumerate(seq):
            key = (job, op_id)
            if key in original_start:
                dev = pulp.LpVariable(f"dev_{j}_{o}", lowBound=0, cat=var_cat)
                deviation_vars[(j, o)] = dev
                prob += dev >= starts[(j, o)] - original_start[key]
                prob += dev >= original_start[key] - starts[(j, o)]

    prob += r * pulp.lpSum(abs_lateness.values()) + (1 - r) * pulp.lpSum(deviation_vars.values())

    # 7. Technologische Reihenfolge & Startrestriktionen
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        earliest = max(arrival[job], last_executed_end.get(job, reschedule_start))
        prob += starts[(j, 0)] >= earliest
        for o in range(1, len(seq)):
            prob += starts[(j, o)] >= starts[(j, o - 1)] + seq[o - 1][2]

    # 8. Maschinenkonflikte inkl. Fixblöcke
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

        for j1, o1, d1 in ops_on_m:
            for fixed_start, fixed_end, _ in fixed_ops.get(m, []):
                y_fix = pulp.LpVariable(f"y_fix_{j1}_{o1}_{int(fixed_start)}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= fixed_start + bigM * (1 - y_fix)
                prob += fixed_end + epsilon <= starts[(j1, o1)] + bigM * y_fix

    # 9. Solver starten
    solver_args.setdefault("msg", True)
    solver_args.setdefault("timeLimit", time_limit)
    solver = solver.upper()
    if solver == "HIGHS":
        cmd = pulp.HiGHS_CMD(**solver_args)
    elif solver == "CBC":
        cmd = pulp.PULP_CBC_CMD(**solver_args)
    else:
        raise ValueError("Solver must be 'CBC' or 'HiGHS'")
    prob.solve(cmd)
    objective_value = pulp.value(prob.objective)

    # 10. Ergebnisse extrahieren
    df_schedule = get_schedule_df(jobs, all_ops, starts, df_jssp, df_times, job_column)
    df_schedule["Lateness"] = (df_schedule["End"] - df_schedule["Deadline"]).clip(lower=0).round(2)
    df_schedule["Absolute Lateness"] = df_schedule["Lateness"].abs()

    # 11. Logging
    print("\nSolver-Informationen:")
    print(f"  Zielfunktionswert         : {round(objective_value, 4)}")
    print(f"  Solver-Status             : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen          : {len(prob.variables())}")
    print(f"  Anzahl Constraints        : {len(prob.constraints)}")
    print(f"  Laufzeit                  : ~{time.time() - start_time:.0f} Sekunden")

    return df_schedule


# einfach (nur fixierte Opertion, die hineinlaufen)
def solve_jssp_sum_with_fixed_ops(df_jssp: pd.DataFrame, df_times: pd.DataFrame, df_executed: pd.DataFrame,
                                                    reschedule_start: float = 1440.0, job_column: str = "Job", solver: str = "HiGHS",
                                                    epsilon: float = 0.0, var_cat: str = "Continuous", time_limit: int | None = 10800,
                                                    sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert die Summe der absoluten Lateness unter Berücksichtigung bereits ausgeführter Operationen.

    Rückgabe:
    - DataFrame mit ['Job','Operation','Arrival','Deadline','Machine',
      'Start','Processing Time','End','Lateness','Absolute Lateness'] (+ optional Production_Plan_ID)
    """
    start_time = time.time()

    # 1. Vorverarbeitung: Zeiten, Sortierung, Jobs extrahieren
    df_times = df_times.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_times.set_index(job_column)["Arrival"].to_dict()
    deadline = df_times.set_index(job_column)["Deadline"].to_dict()
    jobs = df_times[job_column].tolist()

    # 2. BigM berechnen
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_deadline = max(deadline.values())
    num_machines = df_jssp["Machine"].nunique()
    bigM = math.ceil((max_deadline - min_arrival + sum_proc_time / math.sqrt(num_machines)) / 1000) * 1000
    print(f"BigM: {bigM}")

    # 3. Operationen je Job gruppieren
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

    n = len(jobs)

    # 4. Fixierte Operationen ermitteln
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    fixed_ops = {
        m: list(grp[["Start", "End", job_column]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby(job_column)["End"].max().to_dict()

    # 5. Optimierungsmodell aufbauen
    prob = pulp.LpProblem("JSSP_SumAbsLateness_FixedOps", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]], cat=var_cat)
        for j in range(n)
        for o in range(len(all_ops[j]))
    }
    ends = {
        j: pulp.LpVariable(f"end_{j}", lowBound=arrival[jobs[j]], cat=var_cat)
        for j in range(n)
    }
    abs_lateness = {
        j: pulp.LpVariable(f"abs_lateness_{j}", lowBound=0, cat=var_cat)
        for j in range(n)
    }

    # Zielfunktion: Summe der absoluten Lateness
    prob += pulp.lpSum(abs_lateness[j] for j in range(n))

    # 6. Technologische Reihenfolge & Lateness
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        earliest = max(arrival[job], last_executed_end.get(job, reschedule_start))
        prob += starts[(j, 0)] >= earliest
        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev
        d_last = seq[-1][2]
        prob += ends[j] == starts[(j, len(seq) - 1)] + d_last
        lateness = ends[j] - deadline[job]
        prob += abs_lateness[j] >= lateness
        prob += abs_lateness[j] >= -lateness

    # 7. Maschinenkonflikte inkl. fixierter Blöcke
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

        for j1, o1, d1 in ops_on_m:
            for fixed_start, fixed_end, _ in fixed_ops.get(m, []):
                y_fix = pulp.LpVariable(f"y_fix_{j1}_{o1}_{int(fixed_start)}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= fixed_start + bigM * (1 - y_fix)
                prob += fixed_end + epsilon <= starts[(j1, o1)] + bigM * y_fix

    # 8. Solver konfigurieren und aufrufen
    solver_args.setdefault("msg", True)
    solver_args.setdefault("timeLimit", time_limit)
    solver = solver.upper()
    if solver == "HIGHS":
        cmd = pulp.HiGHS_CMD(**solver_args)
    elif solver == "CBC":
        cmd = pulp.PULP_CBC_CMD(**solver_args)
    else:
        raise ValueError("Solver must be 'CBC' or 'HiGHS'")
    prob.solve(cmd)
    objective_value = pulp.value(prob.objective)

    # 9. Ergebnisse extrahieren
    df_schedule = get_schedule_df(jobs, all_ops, starts, df_jssp, df_times, job_column)
    df_schedule["Lateness"] = (df_schedule["End"] - df_schedule["Deadline"]).clip(lower=0).round(2)
    df_schedule["Absolute Lateness"] = df_schedule["Lateness"].abs()

    # 10. Logging
    print("\nSolver-Informationen:")
    print(f"  Summe absolute Lateness  : {round(objective_value, 4)}")
    print(f"  Solver-Status            : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen         : {len(prob.variables())}")
    print(f"  Anzahl Constraints       : {len(prob.constraints)}")
    print(f"  Laufzeit                 : ~{time.time() - start_time:.0f} Sekunden")

    return df_schedule
