import math
import pulp
import pandas as pd

# Tardiness Rescheduling with Arrivals & Deadline -------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------

# Min. Summe Tardiness -------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# mit Deviation Penalty ----------------
def solve_jssp_sum_tardiness_with_devpen(df_jssp: pd.DataFrame, df_arrivals_deadlines: pd.DataFrame, df_executed: pd.DataFrame, 
                                         df_original_plan: pd.DataFrame, r: float = 0.5, reschedule_start: float = 1440.0,
                                         solver: str = 'HiGHS', epsilon: float = 0.0, sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert eine bikriterielle Zielfunktion: Summe der Tardiness und Abweichung vom ursprünglichen Plan.

    Bereits ausgeführte Operationen (aus df_executed) bleiben erhalten. Neue Operationen werden ab
    reschedule_start neu geplant. Maschinenkonflikte und technologische Reihenfolge
    werden berücksichtigt.

    Zielfunktion: Z(σ) = r * T(σ) + (1 - r) * D(σ)
    """

    # Vorbereitung
    df_arrivals_deadlines = df_arrivals_deadlines.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals_deadlines.set_index("Job")["Arrival"].to_dict()
    deadline = df_arrivals_deadlines.set_index("Job")["Deadline"].to_dict()
    jobs = df_arrivals_deadlines["Job"].tolist()

    original_start = {
        (row["Job"], row["Operation"]): row["Start"]
        for _, row in df_original_plan.iterrows()
    }

    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_deadline = max(deadline.values())
    num_machines = df_jssp["Machine"].nunique()
    bigM = math.ceil((max_deadline - min_arrival + sum_proc_time / math.sqrt(num_machines)) / 1000) * 1000

    ops_grouped = df_jssp.sort_values(["Job", "Operation"]).groupby("Job")
    all_ops, machines = [], set()
    for job in jobs:
        seq = []
        for _, row in ops_grouped.get_group(job).iterrows():
            op_id, m, d = row["Operation"], str(row["Machine"]), float(row["Processing Time"])
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    n = len(jobs)
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    fixed_ops = {
        m: list(g[["Start", "End", "Job"]].itertuples(index=False, name=None))
        for m, g in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby("Job")["End"].max().to_dict()

    # Modell
    prob = pulp.LpProblem("JSSP_Tardiness_Deviation", pulp.LpMinimize)

    starts = {(j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]])
              for j in range(n)
              for o in range(len(all_ops[j]))}
    ends = {j: pulp.LpVariable(f"end_{j}", lowBound=arrival[jobs[j]]) for j in range(n)}
    tard = {j: pulp.LpVariable(f"tardiness_{j}", lowBound=0) for j in range(n)}
    deviation_vars = {}

    # Ziel: Tardiness und Abweichung
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        d_last = seq[-1][2]
        prob += ends[j] == starts[(j, len(seq)-1)] + d_last
        prob += tard[j] >= ends[j] - deadline[job]

        for o, (op_id, _, _) in enumerate(seq):
            key = (job, op_id)
            if key in original_start:
                dev = pulp.LpVariable(f"dev_{j}_{o}", lowBound=0)
                deviation_vars[(j, o)] = dev
                prob += dev >= starts[(j, o)] - original_start[key]
                prob += dev >= original_start[key] - starts[(j, o)]

    prob += r * pulp.lpSum(tard.values()) + (1 - r) * pulp.lpSum(deviation_vars.values())

    # Technologische Reihenfolge und Startrestriktionen
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        earliest = max(arrival[job], last_executed_end.get(job, reschedule_start))
        prob += starts[(j, 0)] >= earliest
        for o in range(1, len(seq)):
            prob += starts[(j, o)] >= starts[(j, o - 1)] + seq[o - 1][2]

    # Maschinenkonflikte inkl. Fixierte
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

    # Solver-Auswahl
    solver_args.setdefault("msg", True)
    solver = solver.upper()
    if solver == "HIGHS":
        cmd = pulp.HiGHS_CMD(**solver_args)
    elif solver == "CBC":
        cmd = pulp.PULP_CBC_CMD(**solver_args)
    else:
        raise ValueError("Solver must be 'CBC' or 'HiGHS'")

    prob.solve(cmd)
    objective_value = pulp.value(prob.objective)

    # Ergebnisse
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            ed = st + d
            records.append({
                "Job": job,
                "Operation": op_id,
                "Arrival": arrival[job],
                "Deadline": deadline[job],
                "Machine": m,
                "Start": round(st, 2),
                "Processing Time": d,
                "End": round(ed, 2),
                "Tardiness": max(0, round(ed - deadline[job], 2))
            })

    df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", "Job", "Operation"]).reset_index(drop=True)

    print("\nSolver-Informationen:")
    print(f"  Zielfunktionswert       : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")

    return df_schedule


# einfach ----------------
def solve_jssp_sum_tardiness_with_fixed_ops(df_jssp: pd.DataFrame, df_arrivals_deadlines: pd.DataFrame, df_executed: pd.DataFrame,
                                            reschedule_start: float = 1440.0, solver: str = 'HiGHS', epsilon: float = 0.0,
                                            sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert die Summe der Tardiness (Verspätungen) aller Jobs mit fixierten Operationen.

    Bereits ausgeführte Operationen (aus df_executed) bleiben erhalten. Neue Operationen werden ab
    reschedule_start neu geplant. Maschinenkonflikte und technologische Reihenfolge
    werden vollständig berücksichtigt.

    Zielfunktion: sum_j [ max(0, Endzeit_j - Deadline_j) ]
    """

    # 1. Vorverarbeitung
    df_arrivals_deadlines = df_arrivals_deadlines.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals_deadlines.set_index("Job")["Arrival"].to_dict()
    deadline = df_arrivals_deadlines.set_index("Job")["Deadline"].to_dict()
    jobs = df_arrivals_deadlines["Job"].tolist()

    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_deadline = max(deadline.values())
    num_machines = df_jssp["Machine"].nunique()
    bigM = math.ceil((max_deadline - min_arrival + sum_proc_time / math.sqrt(num_machines)) / 1000) * 1000

    # 2. Operationen je Job
    ops_grouped = df_jssp.sort_values(["Job", "Operation"]).groupby("Job")
    all_ops = []
    machines = set()
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

    # Fixierte Operationen
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    fixed_ops = {
        m: list(grp[["Start", "End", "Job"]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby("Job")["End"].max().to_dict()

    # 3. Modell
    prob = pulp.LpProblem("JSSP_SumTardiness_Fixed", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]])
        for j in range(n)
        for o in range(len(all_ops[j]))
    }

    ends = {
        j: pulp.LpVariable(f"end_{j}", lowBound=arrival[jobs[j]])
        for j in range(n)
    }

    tard = {
        j: pulp.LpVariable(f"tardiness_{j}", lowBound=0)
        for j in range(n)
    }

    # Zielfunktion
    prob += pulp.lpSum(tard[j] for j in range(n))

    # Technologische Reihenfolge + Tardiness
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

    # Maschinenkonflikte inkl. Fixierte
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

    # Solver
    solver_args.setdefault("msg", True)
    solver = solver.upper()
    if solver == "HIGHS":
        cmd = pulp.HiGHS_CMD(**solver_args)
    elif solver == "CBC":
        cmd = pulp.PULP_CBC_CMD(**solver_args)
    else:
        raise ValueError("Solver must be 'CBC' or 'HiGHS'")

    prob.solve(cmd)
    objective_value = pulp.value(prob.objective)

    # Ergebnisse extrahieren
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            ed = st + d
            records.append({
                "Job": job,
                "Operation": op_id,
                "Arrival": arrival[job],
                "Deadline": deadline[job],
                "Machine": m,
                "Start": round(st, 2),
                "Processing Time": d,
                "End": round(ed, 2),
                "Tardiness": max(0, round(ed - deadline[job], 2))
            })

    df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", "Job", "Operation"]).reset_index(drop=True)

    # Logging
    print("\nSolver-Informationen:")
    print(f"  Summe Tardiness         : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")

    return df_schedule


# Min. Max Tardiness ---------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# mit Deviation Penalty ----------------
def solve_jssp_max_tardiness_with_devpen(df_jssp: pd.DataFrame, df_arrivals_deadlines: pd.DataFrame, df_executed: pd.DataFrame,
                                         df_original_plan: pd.DataFrame, r: float = 0.5, reschedule_start: float = 1440.0,
                                         solver: str = 'HiGHS', epsilon: float = 0.0, sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert eine bikriterielle Zielfunktion:
    Maximale Tardiness + Abweichung vom ursprünglichen Plan (weighted sum).

    Zielfunktion: Z(σ) = r * max_j Tardiness_j + (1 - r) * D(σ)
    """

    # 1. Vorverarbeitung
    df_arrivals_deadlines = df_arrivals_deadlines.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals_deadlines.set_index("Job")["Arrival"].to_dict()
    deadline = df_arrivals_deadlines.set_index("Job")["Deadline"].to_dict()
    jobs = df_arrivals_deadlines["Job"].tolist()

    original_start = {
        (row["Job"], row["Operation"]): row["Start"]
        for _, row in df_original_plan.iterrows()
    }

    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_deadline = max(deadline.values())
    num_machines = df_jssp["Machine"].nunique()
    bigM = math.ceil((max_deadline - min_arrival + sum_proc_time / math.sqrt(num_machines)) / 1000) * 1000

    # 2. Operationen je Job
    ops_grouped = df_jssp.sort_values(["Job", "Operation"]).groupby("Job")
    all_ops = []
    machines = set()
    for job in jobs:
        seq = []
        for _, row in ops_grouped.get_group(job).iterrows():
            op_id, m, d = row["Operation"], str(row["Machine"]), float(row["Processing Time"])
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    n = len(jobs)

    # Fixierte Operationen
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    fixed_ops = {
        m: list(grp[["Start", "End", "Job"]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby("Job")["End"].max().to_dict()

    # 3. Modell
    prob = pulp.LpProblem("JSSP_MaxTardiness_DevPen", pulp.LpMinimize)

    starts = {(j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]])
              for j in range(n)
              for o in range(len(all_ops[j]))}

    ends = {j: pulp.LpVariable(f"end_{j}", lowBound=arrival[jobs[j]]) for j in range(n)}
    tard = {j: pulp.LpVariable(f"tardiness_{j}", lowBound=0) for j in range(n)}
    max_tard = pulp.LpVariable("max_tardiness", lowBound=0)

    deviation_vars = {}
    for j, job in enumerate(jobs):
        for o, (op_id, _, _) in enumerate(all_ops[j]):
            key = (job, op_id)
            if key in original_start:
                dev = pulp.LpVariable(f"dev_{j}_{o}", lowBound=0)
                deviation_vars[(j, o)] = dev
                prob += dev >= starts[(j, o)] - original_start[key]
                prob += dev >= original_start[key] - starts[(j, o)]

    # Zielfunktion
    prob += r * max_tard + (1 - r) * pulp.lpSum(deviation_vars.values())

    # 4. Technologische Reihenfolge & Tardiness
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

    # 5. Maschinenkonflikte inkl. Fixierte
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

    # 6. Solver
    solver_args.setdefault("msg", True)
    solver = solver.upper()
    if solver == "HIGHS":
        cmd = pulp.HiGHS_CMD(**solver_args)
    elif solver == "CBC":
        cmd = pulp.PULP_CBC_CMD(**solver_args)
    else:
        raise ValueError("Solver must be 'CBC' or 'HiGHS'")

    prob.solve(cmd)
    objective_value = pulp.value(prob.objective)

    # 7. Ergebnis extrahieren
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            ed = st + d
            records.append({
                "Job": job,
                "Operation": op_id,
                "Arrival": arrival[job],
                "Deadline": deadline[job],
                "Machine": m,
                "Start": round(st, 2),
                "Processing Time": d,
                "End": round(ed, 2),
                "Tardiness": max(0, round(ed - deadline[job], 2))
            })

    df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", "Job", "Operation"]).reset_index(drop=True)

    print("\nSolver-Informationen:")
    print(f"  Zielfunktionswert       : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")

    return df_schedule


# einfach ----------------
def solve_jssp_max_tardiness_with_fixed_ops(df_jssp: pd.DataFrame, df_arrivals_deadlines: pd.DataFrame, df_executed: pd.DataFrame,
                                            reschedule_start: float = 1440.0, solver: str = 'HiGHS', epsilon: float = 0.0,
                                            sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert die maximale Tardiness unter allen Jobs mit fixierten Operationen.
    """

    # 1. Vorverarbeitung
    df_arrivals_deadlines = df_arrivals_deadlines.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals_deadlines.set_index("Job")["Arrival"].to_dict()
    deadline = df_arrivals_deadlines.set_index("Job")["Deadline"].to_dict()
    jobs = df_arrivals_deadlines["Job"].tolist()

    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_deadline = max(deadline.values())
    num_machines = df_jssp["Machine"].nunique()
    bigM = math.ceil((max_deadline - min_arrival + sum_proc_time / math.sqrt(num_machines)) / 1000) * 1000

    # 2. Operationen je Job
    ops_grouped = df_jssp.sort_values(["Job", "Operation"]).groupby("Job")
    all_ops = []
    machines = set()
    for job in jobs:
        seq = []
        for _, row in ops_grouped.get_group(job).iterrows():
            op_id, m, d = row["Operation"], str(row["Machine"]), float(row["Processing Time"])
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    n = len(jobs)

    # Fixierte Operationen aus df_executed
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    fixed_ops = {
        m: list(grp[["Start", "End", "Job"]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby("Job")["End"].max().to_dict()

    # 3. LP-Modell
    prob = pulp.LpProblem("JSSP_MaxTardiness_Fixed", pulp.LpMinimize)

    starts = {(j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]])
              for j in range(n)
              for o in range(len(all_ops[j]))}

    ends = {j: pulp.LpVariable(f"end_{j}", lowBound=arrival[jobs[j]]) for j in range(n)}
    tard = {j: pulp.LpVariable(f"tardiness_{j}", lowBound=0) for j in range(n)}
    max_tard = pulp.LpVariable("max_tardiness", lowBound=0)

    # Zielfunktion
    prob += max_tard

    # 4. Technologische Reihenfolge & Tardiness-Berechnung
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

    # 5. Maschinenkonflikte (inkl. Fixierte)
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

    # 6. Solver
    solver_args.setdefault("msg", True)
    solver = solver.upper()
    if solver == "HIGHS":
        cmd = pulp.HiGHS_CMD(**solver_args)
    elif solver == "CBC":
        cmd = pulp.PULP_CBC_CMD(**solver_args)
    else:
        raise ValueError("Solver must be 'CBC' or 'HiGHS'")

    prob.solve(cmd)
    objective_value = pulp.value(prob.objective)

    # 7. Ergebnis extrahieren
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            ed = st + d
            records.append({
                "Job": job,
                "Operation": op_id,
                "Arrival": arrival[job],
                "Deadline": deadline[job],
                "Machine": m,
                "Start": round(st, 2),
                "Processing Time": d,
                "End": round(ed, 2),
                "Tardiness": max(0, round(ed - deadline[job], 2))
            })

    df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", "Job", "Operation"]).reset_index(drop=True)

    # 8. Logging
    print("\nSolver-Informationen:")
    print(f"  Maximale Tardiness      : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")

    return df_schedule



# Tardiness Scheduling with Arrivals & Deadline ---------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# - Summe
# - Max


# Min. Summe Tardiness ------------------------------------------------------------------------------------------------
def solve_jssp_sum_tardiness(df_jssp: pd.DataFrame, df_arrivals_deadlines: pd.DataFrame, solver: str = 'HiGHS', 
                             epsilon: float = 0.0, sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert die Summe der Tardiness (Verspätungen) aller Jobs.
    Zielfunktion: sum_j [ max(0, Endzeit_j - Deadline_j) ]

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - df_arrivals_deadlines: DataFrame mit ['Job','Arrival','Deadline'].
    - solver: 'CBC' oder 'HiGHS' (case-insensitive).
    - epsilon: Pufferzeit zwischen Operationen auf derselben Maschine.
    - sort_ascending: Sortiert Jobs nach Deadline.
    - **solver_args: Weitere Solver-Parameter wie msg=True, timeLimit=1200 etc.

    Rückgabe:
    - DataFrame mit ['Job','Operation','Arrival','Deadline','Machine',
      'Start','Processing Time','End','Tardiness'].
    """

    # 1. Vorverarbeitung
    df_arrivals_deadlines = df_arrivals_deadlines.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals_deadlines.set_index("Job")["Arrival"].to_dict()
    deadline = df_arrivals_deadlines.set_index("Job")["Deadline"].to_dict()
    jobs = df_arrivals_deadlines["Job"].tolist()

    # BigM berechnen (Worst Case)
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_deadline = max(deadline.values())
    num_machines = df_jssp["Machine"].nunique()

    bigM_raw = max_deadline - min_arrival + sum_proc_time / math.sqrt(num_machines)
    bigM = math.ceil(bigM_raw / 1000) * 1000
    print(f"BigM: {bigM}")

    # 2. Operationen je Job
    ops_grouped = df_jssp.sort_values(["Job", "Operation"]).groupby("Job")
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

    # 3. Modell
    prob = pulp.LpProblem("JSSP_SumTardiness", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]])
        for j in range(n)
        for o in range(len(all_ops[j]))
    }

    ends = {
        j: pulp.LpVariable(f"end_{j}", lowBound=arrival[jobs[j]])
        for j in range(n)
    }

    tard = {
        j: pulp.LpVariable(f"tardiness_{j}", lowBound=0)
        for j in range(n)
    }

    # 4. Zielfunktion
    prob += pulp.lpSum(tard[j] for j in range(n))

    # 5. Technologische Reihenfolge & Tardiness
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev
        d_last = seq[-1][2]
        prob += ends[j] == starts[(j, len(seq) - 1)] + d_last
        prob += tard[j] >= ends[j] - deadline[job]

    # 6. Maschinenkonflikte
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

    # 7. Solverwahl
    solver_args.setdefault("msg", True)
    solver = solver.upper()
    if solver == "HIGHS":
        cmd = pulp.HiGHS_CMD(**solver_args)
    elif solver == "CBC":
        cmd = pulp.PULP_CBC_CMD(**solver_args)
    else:
        raise ValueError("Solver must be 'CBC' or 'HiGHS'")
    
    prob.solve(cmd)
    objective_value = pulp.value(prob.objective)

    # 8. Ergebnisse
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            ed = st + d
            records.append({
                "Job": job,
                "Operation": op_id,
                "Arrival": arrival[job],
                "Deadline": deadline[job],
                "Machine": m,
                "Start": round(st, 2),
                "Processing Time": d,
                "End": round(ed, 2),
                "Tardiness": max(0, round(ed - deadline[job], 2))
            })

    df_schedule = (
        pd.DataFrame.from_records(records)
        .sort_values(["Start", "Job", "Operation"])
        .reset_index(drop=True)
    )

    # 9. Logging
    print("\nSolver-Informationen:")
    print(f"  Summe Tardiness         : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")

    return df_schedule


# Min. Max Tardiness --------------------------------------------------------------------------------------------------
def solve_jssp_max_tardiness(df_jssp: pd.DataFrame, df_arrivals_deadlines: pd.DataFrame, solver: str = 'HiGHS', 
                             epsilon: float = 0.0, sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert die maximale Tardiness (Verspätung) unter allen Jobs.
    Zielfunktion: max_j [ max(0, Endzeit_j - Deadline_j) ]

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - df_arrivals_deadlines: DataFrame mit ['Job','Arrival','Deadline'].
    - solver: 'CBC' oder 'HiGHS' (case-insensitive).
    - epsilon: Pufferzeit zwischen Operationen auf derselben Maschine.
    - sort_ascending: Sortiert die Jobs nach Deadline.
    - **solver_args: Weitere Solver-Parameter wie msg=True, timeLimit=1200 etc.

    Rückgabe:
    - DataFrame mit ['Job','Operation','Arrival','Deadline','Machine',
      'Start','Processing Time','End','Tardiness'].
    """

    # 1. Vorverarbeitung
    df_arrivals_deadlines = df_arrivals_deadlines.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals_deadlines.set_index("Job")["Arrival"].to_dict()
    deadline = df_arrivals_deadlines.set_index("Job")["Deadline"].to_dict()
    jobs = df_arrivals_deadlines["Job"].tolist()

    # BigM berechnen (Worst Case)
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_deadline = max(deadline.values())
    num_machines = df_jssp["Machine"].nunique()
    
    bigM_raw = max_deadline - min_arrival + sum_proc_time / math.sqrt(num_machines)
    bigM = math.ceil(bigM_raw / 1000) * 1000
    print(f"BigM: {bigM}")

    # 2. Operationen je Job
    ops_grouped = df_jssp.sort_values(["Job", "Operation"]).groupby("Job")
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

    # 3. LP-Modell
    prob = pulp.LpProblem("JSSP_MaxTardiness", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]])
        for j in range(n)
        for o in range(len(all_ops[j]))
    }

    ends = {
        j: pulp.LpVariable(f"end_{j}", lowBound=arrival[jobs[j]])
        for j in range(n)
    }

    tard = {
        j: pulp.LpVariable(f"tardiness_{j}", lowBound=0)
        for j in range(n)
    }

    max_tard = pulp.LpVariable("max_tardiness", lowBound=0)

    # Zielfunktion
    prob += max_tard

    # 4. Technologische Reihenfolge & Tardiness
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev
        d_last = seq[-1][2]
        prob += ends[j] == starts[(j, len(seq) - 1)] + d_last
        prob += tard[j] >= ends[j] - deadline[job]
        prob += max_tard >= tard[j]

    # 5. Maschinenkonflikte
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

    # 6. Solverwahl
    solver_args.setdefault("msg", True)
    solver = solver.upper()
    if solver == "HIGHS":
        cmd = pulp.HiGHS_CMD(**solver_args)
    elif solver == "CBC":
        cmd = pulp.PULP_CBC_CMD(**solver_args)
    else:
        raise ValueError("Solver must be 'CBC' or 'HiGHS'")

    prob.solve(cmd)
    objective_value = pulp.value(prob.objective)

    # 7. Ergebnis extrahieren
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            ed = st + d
            records.append({
                "Job": job,
                "Operation": op_id,
                "Arrival": arrival[job],
                "Deadline": deadline[job],
                "Machine": m,
                "Start": round(st, 2),
                "Processing Time": d,
                "End": round(ed, 2),
                "Tardiness": max(0, round(ed - deadline[job], 2))
            })

    df_schedule = (
        pd.DataFrame.from_records(records)
        .sort_values(["Start", "Job", "Operation"])
        .reset_index(drop=True)
    )

    # 8. Logging
    print("\nSolver-Informationen:")
    print(f"  Maximale Tardiness      : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")

    return df_schedule

    

# FCFS --------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------

def schedule_fcfs_with_arrivals(df_jssp: pd.DataFrame,
                                arrival_df: pd.DataFrame) -> pd.DataFrame:
    """
    FCFS-Scheduling mit Job-Ankunftszeiten auf Basis eines DataFrames.

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - arrival_df: DataFrame mit ['Job','Arrival'].
    """
    # Arrival-Zeiten als Dict
    arrival = arrival_df.set_index('Job')['Arrival'].to_dict()

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
            # Skip, wenn alle Ops geplant
            if op_idx >= (df_jssp['Job'] == job).sum():
                continue
            # Hole Row anhand Job+Operation
            row = df_jssp[(df_jssp['Job']==job)&(df_jssp['Operation']==op_idx)].iloc[0]
            m = int(row['Machine'].lstrip('M'))
            dur = row['Processing Time']
            earliest = max(job_ready[job], machine_ready[m])
            # Best-Kandidat wählen
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
        # Update Status
        job_ready[job] = end
        machine_ready[m] = end
        next_op[job] += 1
        remaining -= 1

    df_schedule = pd.DataFrame(schedule)
    return df_schedule.sort_values(['Arrival','Start'])
    