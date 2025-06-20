import math
import pulp
import pandas as pd
import time

# Lateness Scheduling -----------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# - Summe Absolute Lateness
# - Max Absolute Lateness


# Min. Summe Absolute Lateness ----------------------------------------------------------------------------------------

def solve_jssp_sum(df_jssp: pd.DataFrame, df_times: pd.DataFrame, job_column: str = "Job",
                                     solver: str = 'HiGHS', epsilon: float = 0.0, var_cat: str = "Continuous", 
                                     time_limit: int | None = 10800, sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert die Summe der absoluten Lateness (Früh- oder Spätfertigung) aller Jobs.
    Zielfunktion: sum_j [ |C_j - d_j| ]

    Rückgabe:
    - DataFrame mit ['Job','Operation','Arrival','Deadline','Machine',
      'Start','Processing Time','End','Lateness','Absolute Lateness'] (+ optional Production_Plan_ID)
    """
    start_time = time.time()

    # 1. Vorverarbeitung: Zeiten und Sortierung
    df_times = df_times.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_times.set_index(job_column)["Arrival"].to_dict()
    deadline = df_times.set_index(job_column)["Deadline"].to_dict()
    jobs = df_times[job_column].tolist()

    # 2. BigM berechnen für Konflikte
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_deadline = max(deadline.values())
    num_machines = df_jssp["Machine"].nunique()
    bigM_raw = max_deadline - min_arrival + sum_proc_time / math.sqrt(num_machines)
    bigM = math.ceil(bigM_raw / 1000) * 1000
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

    # 4. LP-Modell erstellen
    prob = pulp.LpProblem("JSSP_SumAbsoluteLateness", pulp.LpMinimize)

    # 5. Variablen definieren
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

    # Zielfunktion: Summe der absoluten Abweichungen
    prob += pulp.lpSum(abs_lateness[j] for j in range(n))

    # 6. Technologische Reihenfolge & Lateness-Constraints
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev

        d_last = seq[-1][2]
        prob += ends[j] == starts[(j, len(seq) - 1)] + d_last

        lateness = ends[j] - deadline[job]
        prob += abs_lateness[j] >= lateness
        prob += abs_lateness[j] >= -lateness

    # 7. Maschinenkonflikte verhindern
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

    # 8. Solver konfigurieren & aufrufen
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
    records = get_records(jobs=jobs, all_ops=all_ops, 
                          starts=starts, arrival=arrival, 
                          deadline=deadline, job_column=job_column, df_times=df_times)
    df_schedule = (
        pd.DataFrame.from_records(records)
        .sort_values(["Start", job_column, "Operation"])
        .reset_index(drop=True)
    )

    # 10. Logging & Rückgabe
    print("\nSolver-Informationen:")
    print(f"  Summe absolute Lateness  : {round(objective_value, 4)}")
    print(f"  Solver-Status            : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen         : {len(prob.variables())}")
    print(f"  Anzahl Constraints       : {len(prob.constraints)}")
    print(f"  Laufzeit                 : ~{time.time() - start_time:.0f} Sekunden")

    return df_schedule


# Min. Max Absolute Latenesss -----------------------------------------------------------------------------------------
def solve_jssp_max(df_jssp: pd.DataFrame, df_times: pd.DataFrame, job_column: str = "Job",
                                     solver: str = "HiGHS", epsilon: float = 0.0, var_cat: str = "Continuous",
                                     time_limit: int | None = 10800, sort_ascending: bool = False,  **solver_args) -> pd.DataFrame:
    """
    Minimiert die maximale absolute Lateness (Früh- oder Spätfertigung) über alle Jobs.
    Zielfunktion: min max_j [ |C_j - d_j| ]
    """
    import time
    start_time = time.time()

    # 1. Vorverarbeitung: Ankunftszeiten, Deadlines, Jobliste
    df_times = df_times.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_times.set_index(job_column)["Arrival"].to_dict()
    deadline = df_times.set_index(job_column)["Deadline"].to_dict()
    jobs = df_times[job_column].tolist()

    # 2. BigM berechnen zur Konfliktauflösung
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_deadline = max(deadline.values())
    num_machines = df_jssp["Machine"].nunique()
    bigM_raw = max_deadline - min_arrival + sum_proc_time / math.sqrt(num_machines)
    bigM = math.ceil(bigM_raw / 1000) * 1000
    print(f"BigM: {bigM}")

    # 3. Operationen je Job aufbauen
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

    # 4. Modell initialisieren
    prob = pulp.LpProblem("JSSP_MaxAbsLateness", pulp.LpMinimize)

    # 5. Variablen: Startzeit, Endzeit, Lateness, Maximale Lateness
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
    max_abs_lateness = pulp.LpVariable("max_abs_lateness", lowBound=0, cat=var_cat)

    # Zielfunktion: Minimierung der maximalen absoluten Lateness
    prob += max_abs_lateness

    # 6. Technologische Reihenfolge und Lateness-Constraints
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev
        d_last = seq[-1][2]
        prob += ends[j] == starts[(j, len(seq) - 1)] + d_last
        lateness = ends[j] - deadline[job]
        prob += abs_lateness[j] >= lateness
        prob += abs_lateness[j] >= -lateness
        prob += max_abs_lateness >= abs_lateness[j]

    # 7. Maschinenkonflikte: Keine Überlappung
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

    # 8. Solverwahl und Aufruf
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

    # 9. Ergebnisse extrahieren und DataFrame aufbauen
    records = get_records(jobs=jobs, all_ops=all_ops, 
                          starts=starts, arrival=arrival, 
                          deadline=deadline, job_column=job_column, df_times=df_times)
    df_schedule = (
        pd.DataFrame.from_records(records)
        .sort_values(["Start", job_column, "Operation"])
        .reset_index(drop=True)
    )

    # 10. Logging und Rückgabe
    print("\nSolver-Informationen:")
    print(f"  Maximale absolute Lateness : {round(objective_value, 4)}")
    print(f"  Solver-Status              : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen           : {len(prob.variables())}")
    print(f"  Anzahl Constraints         : {len(prob.constraints)}")
    print(f"  Laufzeit                   : ~{time.time() - start_time:.0f} Sekunden")

    return df_schedule


def get_records(jobs, all_ops, starts, arrival, deadline, job_column="Job", df_times=None):

    # 1. Optional: Mapping von Job → Production_Plan_ID
    if df_times is not None and "Production_Plan_ID" in df_times.columns:
        job_production_plan = df_times.set_index(job_column)["Production_Plan_ID"].to_dict()
    else:
        job_production_plan = {}

    # 2. Records sammeln
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            ed = st + d
            lateness = round(ed - deadline[job], 2)
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
                "Lateness": lateness,
                "Absolute Lateness": abs(lateness)
            })
            records.append(record)

    return records

