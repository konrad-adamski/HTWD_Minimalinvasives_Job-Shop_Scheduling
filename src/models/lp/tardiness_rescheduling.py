import time
import math
import pulp
import pandas as pd


# Tardiness Rescheduling with Arrivals & Deadline ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------

# Min. Summe Tardiness -------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# mit Deviation Penalty (& fixierte Operation, die hineinlaufen)
def solve_jssp_sum_with_devpen(df_jssp: pd.DataFrame, df_times: pd.DataFrame, df_executed: pd.DataFrame,
                                         df_original_plan: pd.DataFrame, r: float = 0.5, reschedule_start: float = 1440.0,
                                         job_column: str = "Job", solver: str = "HiGHS", epsilon: float = 0.0,
                                         var_cat: str = "Continuous", time_limit: int | None = 10800,
                                         sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert eine bikriterielle Zielfunktion: Summe der Tardiness und Abweichung vom ursprünglichen Plan.

    Bereits ausgeführte Operationen (aus df_executed) bleiben erhalten. Neue Operationen werden ab
    reschedule_start neu geplant. Maschinenkonflikte und technologische Reihenfolge
    werden berücksichtigt.

    Zielfunktion: Z(σ) = r * T(σ) + (1 - r) * D(σ)
    """
    start_time = time.time()

    # 1. Vorverarbeitung: Zeiten, Reihenfolge, Originalplan
    df_times = df_times.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_times.set_index(job_column)["Arrival"].to_dict()
    deadline = df_times.set_index(job_column)["Deadline"].to_dict()
    jobs = df_times[job_column].tolist()

    original_start = {
        (row[job_column], row["Operation"]): row["Start"]
        for _, row in df_original_plan.iterrows()
    }

    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_deadline = max(deadline.values())
    num_machines = df_jssp["Machine"].nunique()
    bigM = math.ceil((max_deadline - min_arrival + sum_proc_time / math.sqrt(num_machines)) / 1000) * 1000
    print(f"BigM: {bigM}")

    # 2. Job-Operationen
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

    # 3. Fixierte Operationen (Maschinenblockierung)
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    fixed_ops = {
        m: list(g[["Start", "End", job_column]].itertuples(index=False, name=None))
        for m, g in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby(job_column)["End"].max().to_dict()

    # 4. LP-Modell aufstellen
    prob = pulp.LpProblem("JSSP_Tardiness_Deviation", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]], cat=var_cat)
        for j in range(n)
        for o in range(len(all_ops[j]))
    }
    ends = {
        j: pulp.LpVariable(f"end_{j}", lowBound=arrival[jobs[j]], cat=var_cat)
        for j in range(n)
    }
    tard = {
        j: pulp.LpVariable(f"tardiness_{j}", lowBound=0, cat=var_cat)
        for j in range(n)
    }

    deviation_vars = {}

    # 5. Zielfunktion + Constraints zur Abweichung
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        d_last = seq[-1][2]
        prob += ends[j] == starts[(j, len(seq)-1)] + d_last
        prob += tard[j] >= ends[j] - deadline[job]

        for o, (op_id, _, _) in enumerate(seq):
            key = (job, op_id)
            if key in original_start:
                dev = pulp.LpVariable(f"dev_{j}_{o}", lowBound=0, cat=var_cat)
                deviation_vars[(j, o)] = dev
                prob += dev >= starts[(j, o)] - original_start[key]
                prob += dev >= original_start[key] - starts[(j, o)]

    # Zielgewichtung: Mischung aus Tardiness und Abweichung
    prob += r * pulp.lpSum(tard.values()) + (1 - r) * pulp.lpSum(deviation_vars.values())

    # 6. Technologische Reihenfolge + Startrestriktionen
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        earliest = max(arrival[job], last_executed_end.get(job, reschedule_start))
        prob += starts[(j, 0)] >= earliest
        for o in range(1, len(seq)):
            prob += starts[(j, o)] >= starts[(j, o - 1)] + seq[o - 1][2]

    # 7. Maschinenkonflikte inkl. Fixe
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
            for fixed_start, fixed_end, fixed_job in fixed_ops.get(m, []):
                y_fix = pulp.LpVariable(f"y_fix_{j1}_{o1}_{fixed_job}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= fixed_start + bigM * (1 - y_fix)
                prob += fixed_end + epsilon <= starts[(j1, o1)] + bigM * y_fix

    # 8. Solver starten
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

    # 9. Ergebnisaufbereitung
    records = get_records(
        jobs, all_ops, starts,
        arrival, deadline,
        job_column=job_column,
        df_times=df_times
    )

    df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", job_column, "Operation"]).reset_index(drop=True)

    # 10. Logging
    print("\nSolver-Informationen:")
    print(f"  Zielfunktionswert       : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")
    print(f"  Laufzeit                : ~{time.time() - start_time:.0f} Sekunden")

    return df_schedule


# einfach (nur fixierte Opertion, die hineinlaufen)
def solve_jssp_sum_with_fixed_ops(df_jssp: pd.DataFrame, df_times: pd.DataFrame, df_executed: pd.DataFrame,
                                            reschedule_start: float = 1440.0, job_column: str = "Job",
                                            solver: str = "HiGHS", epsilon: float = 0.0, var_cat: str = "Continuous",
                                            time_limit: int | None = 10800, sort_ascending: bool = False,
                                            **solver_args) -> pd.DataFrame:
    """
    Minimiert die Summe der Tardiness (Verspätungen) aller Jobs mit fixierten Operationen.

    Bereits ausgeführte Operationen (aus df_executed) bleiben erhalten. Neue Operationen werden ab
    reschedule_start neu geplant. Maschinenkonflikte und technologische Reihenfolge
    werden vollständig berücksichtigt.

    Zielfunktion: sum_j [ max(0, Endzeit_j - Deadline_j) ]
    """
    start_time = time.time()

    # 1. Vorverarbeitung: Zeiten und Jobs extrahieren
    df_times = df_times.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_times.set_index(job_column)["Arrival"].to_dict()
    deadline = df_times.set_index(job_column)["Deadline"].to_dict()
    jobs = df_times[job_column].tolist()

    # 2. Big-M zur Konfliktmodellierung
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_deadline = max(deadline.values())
    num_machines = df_jssp["Machine"].nunique()
    bigM = math.ceil((max_deadline - min_arrival + sum_proc_time / math.sqrt(num_machines)) / 1000) * 1000
    print(f"BigM: {bigM}")

    # 3. Operationen je Job extrahieren
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

    # 4. Fixierte Operationen vorbereiten
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    fixed_ops = {
        m: list(grp[["Start", "End", job_column]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby(job_column)["End"].max().to_dict()

    # 5. Modellierung mit pulp
    prob = pulp.LpProblem("JSSP_SumTardiness_Fixed", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]], cat=var_cat)
        for j in range(n)
        for o in range(len(all_ops[j]))
    }

    ends = {
        j: pulp.LpVariable(f"end_{j}", lowBound=arrival[jobs[j]], cat=var_cat)
        for j in range(n)
    }

    tard = {
        j: pulp.LpVariable(f"tardiness_{j}", lowBound=0, cat=var_cat)
        for j in range(n)
    }

    # Zielfunktion: Minimierung der Gesamttardiness
    prob += pulp.lpSum(tard[j] for j in range(n))

    # 6. Technologische Abfolge und Tardiness-Definition
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        earliest = max(arrival[job], last_executed_end.get(job, reschedule_start))
        prob += starts[(j, 0)] >= earliest
        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev
        d_last = seq[-1][2]
        prob += ends[j] == starts[(j, len(seq) - 1)] + d_last
        prob += tard[j] >= ends[j] - deadline[job]

    # 7. Maschinenkonflikte (inkl. fixierte)
    for m in machines:
        ops_on_m = [
            (j, o, seq[o][2])
            for j, seq in enumerate(all_ops)
            for o in range(len(seq))
            if seq[o][1] == m
        ]

        # Konflikte zwischen modellierten Operationen
        for i in range(len(ops_on_m)):
            j1, o1, d1 = ops_on_m[i]
            for j2, o2, d2 in ops_on_m[i + 1:]:
                if j1 == j2:
                    continue
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM * (1 - y)
                prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM * y

        # Konflikte mit fixierten Operationen
        for j1, o1, d1 in ops_on_m:
            for fixed_start, fixed_end, fixed_job in fixed_ops.get(m, []):
                y_fix = pulp.LpVariable(f"y_fix_{j1}_{o1}_{fixed_job}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= fixed_start + bigM * (1 - y_fix)
                prob += fixed_end + epsilon <= starts[(j1, o1)] + bigM * y_fix

    # 8. Solverauswahl und Lösung
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

    # 9. Ergebnisaufbereitung
    records = get_records(
        jobs, all_ops, starts,
        arrival, deadline,
        job_column=job_column,
        df_times=df_times
    )

    df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", job_column, "Operation"]).reset_index(drop=True)

    # 10. Logging
    solving_duration = time.time() - start_time
    print("\nSolver-Informationen:")
    print(f"  Summe Tardiness         : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")
    print(f"  Laufzeit                : ~{solving_duration:.0f} Sekunden")

    return df_schedule

# Min. Max Tardiness ---------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# mit Deviation Penalty (& fixierte Operation, die hineinlaufen)
def solve_jssp_max_with_devpen(df_jssp: pd.DataFrame, df_times: pd.DataFrame, df_executed: pd.DataFrame,
                                         df_original_plan: pd.DataFrame, r: float = 0.5, reschedule_start: float = 1440.0,
                                         job_column: str = "Job", solver: str = "HiGHS", epsilon: float = 0.0,
                                         var_cat: str = "Continuous", time_limit: int | None = 10800,
                                         sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert eine bikriterielle Zielfunktion:
    Maximale Tardiness + Abweichung vom ursprünglichen Plan (weighted sum).

    Zielfunktion: Z(σ) = r * max_j Tardiness_j + (1 - r) * D(σ)
    """
    import time, math, pulp
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
    
    # 2. Big-M-Berechnung
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_deadline = max(deadline.values())
    num_machines = df_jssp["Machine"].nunique()
    bigM = math.ceil((max_deadline - min_arrival + sum_proc_time / math.sqrt(num_machines)) / 1000) * 1000
    print(f"BigM: {bigM}")

    # 3. Operationen je Job
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

    # 4. Fixierte Operationen
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    fixed_ops = {
        m: list(grp[["Start", "End", job_column]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby(job_column)["End"].max().to_dict()

    # 5. Modell
    prob = pulp.LpProblem("JSSP_MaxTardiness_DevPen", pulp.LpMinimize)

    starts = {(j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]], cat=var_cat)
              for j in range(n)
              for o in range(len(all_ops[j]))}

    ends = {j: pulp.LpVariable(f"end_{j}", lowBound=arrival[jobs[j]], cat=var_cat) for j in range(n)}
    tard = {j: pulp.LpVariable(f"tardiness_{j}", lowBound=0, cat=var_cat) for j in range(n)}
    max_tard = pulp.LpVariable("max_tardiness", lowBound=0, cat=var_cat)

    deviation_vars = {}
    for j, job in enumerate(jobs):
        for o, (op_id, _, _) in enumerate(all_ops[j]):
            key = (job, op_id)
            if key in original_start:
                dev = pulp.LpVariable(f"dev_{j}_{o}", lowBound=0, cat=var_cat)
                deviation_vars[(j, o)] = dev
                prob += dev >= starts[(j, o)] - original_start[key]
                prob += dev >= original_start[key] - starts[(j, o)]

    # Zielfunktion
    prob += r * max_tard + (1 - r) * pulp.lpSum(deviation_vars.values())

    # 6. Technologische Reihenfolge & Tardiness
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        earliest = max(arrival[job], last_executed_end.get(job, reschedule_start))
        prob += starts[(j, 0)] >= earliest
        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev
        d_last = seq[-1][2]
        prob += ends[j] == starts[(j, len(seq) - 1)] + d_last
        prob += tard[j] >= ends[j] - deadline[job]
        prob += max_tard >= tard[j]

    # 7. Maschinenkonflikte inkl. Fixierte
    for m in machines:
        ops_on_m = [(j, o, seq[o][2])
                    for j, seq in enumerate(all_ops)
                    for o in range(len(seq))
                    if seq[o][1] == m]
        for i in range(len(ops_on_m)):
            j1, o1, d1 = ops_on_m[i]
            for j2, o2, d2 in ops_on_m[i + 1:]:
                if j1 == j2:
                    continue
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM * (1 - y)
                prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM * y

        for j1, o1, d1 in ops_on_m:
            for fixed_start, fixed_end, fixed_job in fixed_ops.get(m, []):
                y_fix = pulp.LpVariable(f"y_fix_{j1}_{o1}_{fixed_job}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= fixed_start + bigM * (1 - y_fix)
                prob += fixed_end + epsilon <= starts[(j1, o1)] + bigM * y_fix

    # 8. Solver
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

    # 9. Ergebnis extrahieren
    records = get_records(jobs, all_ops, starts, arrival, deadline, job_column=job_column,df_times=df_times)

    df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", job_column, "Operation"]).reset_index(drop=True)

    # 10. Logging
    print("\nSolver-Informationen:")
    print(f"  Zielfunktionswert       : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")
    print(f"  Laufzeit                : ~{time.time() - start_time:.0f} Sekunden")

    return df_schedule

# einfach (nur fixierte Opertion, die hineinlaufen)
def solve_jssp_max_with_fixed_ops(df_jssp: pd.DataFrame, df_times: pd.DataFrame, df_executed: pd.DataFrame,
                                            reschedule_start: float = 1440.0, job_column: str = "Job",
                                            solver: str = "HiGHS", epsilon: float = 0.0,
                                            var_cat: str = "Continuous", time_limit: int | None = 10800,
                                            sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert die maximale Tardiness unter allen Jobs mit fixierten Operationen.
    """
    import time, math, pulp
    start_time = time.time()

    # 1. Vorverarbeitung
    df_times = df_times.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_times.set_index(job_column)["Arrival"].to_dict()
    deadline = df_times.set_index(job_column)["Deadline"].to_dict()
    jobs = df_times[job_column].tolist()

    # 2. Big-M-Berechnung
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_deadline = max(deadline.values())
    num_machines = df_jssp["Machine"].nunique()
    bigM = math.ceil((max_deadline - min_arrival + sum_proc_time / math.sqrt(num_machines)) / 1000) * 1000
    print(f"BigM: {bigM}")

    # 3. Operationen je Job
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

    # 4. Fixierte Operationen
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    fixed_ops = {
        m: list(grp[["Start", "End", job_column]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby(job_column)["End"].max().to_dict()

    # 5. Modell
    prob = pulp.LpProblem("JSSP_MaxTardiness_Fixed", pulp.LpMinimize)

    starts = {(j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]], cat=var_cat)
              for j in range(n)
              for o in range(len(all_ops[j]))}

    ends = {j: pulp.LpVariable(f"end_{j}", lowBound=arrival[jobs[j]], cat=var_cat) for j in range(n)}
    tard = {j: pulp.LpVariable(f"tardiness_{j}", lowBound=0, cat=var_cat) for j in range(n)}
    max_tard = pulp.LpVariable("max_tardiness", lowBound=0, cat=var_cat)

    # Zielfunktion
    prob += max_tard

    # 6. Technologische Reihenfolge & Tardiness-Berechnung
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        earliest = max(arrival[job], last_executed_end.get(job, reschedule_start))
        prob += starts[(j, 0)] >= earliest
        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev
        d_last = seq[-1][2]
        prob += ends[j] == starts[(j, len(seq) - 1)] + d_last
        prob += tard[j] >= ends[j] - deadline[job]
        prob += max_tard >= tard[j]

    # 7. Maschinenkonflikte inkl. Fixierte
    for m in machines:
        ops_on_m = [(j, o, seq[o][2])
                    for j, seq in enumerate(all_ops)
                    for o in range(len(seq))
                    if seq[o][1] == m]
        for i in range(len(ops_on_m)):
            j1, o1, d1 = ops_on_m[i]
            for j2, o2, d2 in ops_on_m[i + 1:]:
                if j1 == j2:
                    continue
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM * (1 - y)
                prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM * y

        for j1, o1, d1 in ops_on_m:
            for fixed_start, fixed_end, fixed_job in fixed_ops.get(m, []):
                y_fix = pulp.LpVariable(f"y_fix_{j1}_{o1}_{fixed_job}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= fixed_start + bigM * (1 - y_fix)
                prob += fixed_end + epsilon <= starts[(j1, o1)] + bigM * y_fix

    # 8. Solver
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

    # 9. Ergebnisaufbereitung
    records = get_records(jobs, all_ops, starts, arrival, deadline, job_column=job_column, df_times=df_times)
    df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", job_column, "Operation"]).reset_index(drop=True)

    # 10. Logging
    print("\nSolver-Informationen:")
    print(f"  Maximale Tardiness      : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")
    print(f"  Laufzeit                : ~{time.time() - start_time:.0f} Sekunden")

    return df_schedule


# ----------------------------------------------------------------------------------------------------
def get_records(jobs, all_ops, starts, arrival, deadline, job_column="Job", df_times=None):
    
    # Optional: Mapping von Job → Production_Plan_ID
    if df_times is not None and "Production_Plan_ID" in df_times.columns:
        job_production_plan = df_times.set_index(job_column)["Production_Plan_ID"].to_dict()
    else:
        job_production_plan = {}

    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            ed = st + d
            record = {
                job_column: job,
            }
            if job in job_production_plan:
                record["Production_Plan_ID"] = job_production_plan[job]
            record.update({
                "Operation": op_id,
                "Arrival": arrival[job],
                "Deadline": deadline[job],
                "Machine": m,
                "Start": round(st, 2),
                "Processing Time": d,
                "End": round(ed, 2),
                "Tardiness": max(0, round(ed - deadline[job], 2)),
            })
            records.append(record)
    return records

