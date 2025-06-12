import pandas as pd
import math
import pulp

# - solve_jssp_flowtime
# - solve_jssp_flowtime_with_devpen
# - solve_jssp_individual_flowtime_with_fixed_ops
# - solve_jssp_weighted_flowtime_with_fixed_ops


# Flowtime Scheduling with Arrivals ---------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------

def solve_jssp_flowtime(df_jssp: pd.DataFrame, df_arrivals: pd.DataFrame, solver: str = 'HiGHS',
                        epsilon: float = 0.0, sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert die gesamte Flow Time eines Job-Shop-Problems mit Ankunftszeiten.

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - df_arrivals: DataFrame mit ['Job','Arrival'].
    - solver: 'CBC' oder 'HiGHS' (case-insensitive).
    - time_limit: Max. Zeit in Sekunden für den Solver.
    - epsilon: Pufferzeit in Minuten zwischen zwei Jobs auf derselben Maschine.
    - sort_ascending: Sortierung der Jobs nach Ankunft (True = früh zuerst).
    - **solver_args: Weitere Solver-Parameter wie msg=True, timeLimit= 1200, threads=5 etc.

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Machine',
      'Start','Processing Time','Flow time','End'].
    """

    # 1. Vorverarbeitung
    df_arrivals = df_arrivals.sort_values("Arrival", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals.set_index("Job")["Arrival"].to_dict()
    jobs = df_arrivals["Job"].tolist()

    # BigM berechnen (Worst Case)
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_arrival = max(arrival.values())
    bigM_raw = max_arrival - min_arrival + sum_proc_time
    bigM = math.ceil(bigM_raw / 1000) * 1000
    print(f"BigM: {bigM}")

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

    # 3. LP-Modell
    prob = pulp.LpProblem("JSSP_FlowTime_Arrival", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]])
        for j in range(n)
        for o in range(len(all_ops[j]))
    }

    flow_sum = pulp.LpVariable.dicts("flowtime", jobs, lowBound=0)

    # Zielfunktion: Minimierung der Summe aller Flow Times
    prob += pulp.lpSum([flow_sum[job] for job in jobs])

    # 4. Technologische Reihenfolge + Flow-Zuordnung
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev

        # Letzte Operation bestimmt die Endzeit → FlowTime = End - Arrival
        d_last = seq[-1][2]
        prob += flow_sum[job] == starts[(j, len(seq) - 1)] + d_last - arrival[job]

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

    # 6. Solver auswählen
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

    # 7. Ergebnisse extrahieren
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            ed = st + d
            records.append({
                "Job": job,
                "Operation": op_id,
                "Arrival": arrival[job],
                "Machine": m,
                "Start": round(st, 2),
                "Processing Time": d,
                "Flow time": round(ed - arrival[job], 2),
                "End": round(ed, 2)
            })

    df_schedule = (
        pd.DataFrame(records)
        .sort_values(["Start", "Job", "Operation"])
        .reset_index(drop=True)
    )

    # 8. Logging
    print("\nSolver-Informationen:")
    print(f"  Summe Flow Times        : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")

    return df_schedule


# Flowtime Rescheduling with Arrivals & Deviation Penalty  ----------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------

def solve_jssp_flowtime_with_devpen(df_jssp: pd.DataFrame, df_arrivals: pd.DataFrame,
                                                              df_executed: pd.DataFrame, df_original_plan: pd.DataFrame,
                                                              r: float = 0.5, reschedule_start: float = 1440.0, solver: str = 'HiGHS',
                                                              epsilon: float = 0.0, sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert eine bikriterielle Zielfunktion: Flow-Time und Abweichung zum ursprünglichen Plan.

    Bereits ausgeführte Operationen (aus df_executed) bleiben erhalten. Neue Operationen werden ab
    reschedule_start neu geplant. Maschinenkonflikte und technologische Reihenfolge
    werden vollständig berücksichtigt.

    Zielfunktion: Z(σ) = r * F(σ) + (1 - r) * D(σ)
    - F(σ): Summe der Flow-Times
    - D(σ): Abweichung vom Originalplan

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - df_arrivals: DataFrame mit ['Job','Arrival'].
    - df_executed: DataFrame mit ['Job','Machine','Start','End'].
    - df_original_plan: DataFrame mit ['Job','Operation','Start'].
    - r: Gewichtung von Flow-Time gegenüber Stabilität (zwischen 0 und 1).
    - solver: 'CBC' oder 'HiGHS' (case-insensitive).
    - epsilon: Pufferzeit zwischen Operationen auf derselben Maschine.
    - sort_ascending: Sortierung der Ankunftszeiten.
    - reschedule_start: Zeitpunkt ab dem neu geplant werden soll.
    - **solver_args: Weitere Solver-Parameter wie msg=True, timeLimit=1200 etc.

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Machine','Start','Processing Time','Flow time','End'].
    """

    # 1. Vorverarbeitung
    df_arrivals = df_arrivals.sort_values("Arrival", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals.set_index("Job")["Arrival"].to_dict()
    jobs = df_arrivals["Job"].tolist()

    original_start = {
        (row['Job'], row['Operation']): row['Start']
        for _, row in df_original_plan.iterrows()
    }

    # BigM berechnen (Worst Case)
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_arrival = max(arrival.values())
    bigM_raw = max_arrival - min_arrival + sum_proc_time
    bigM = math.ceil(bigM_raw / 1000) * 1000
    print(f"BigM: {bigM}")

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

    # Fixierte Operationen aus df_executed
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    fixed_ops = {
        m: list(grp[["Start", "End", "Job"]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby("Job")["End"].max().to_dict()

    # 3. LP-Modell
    prob = pulp.LpProblem("JSSP_Bicriteria", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]])
        for j in range(n)
        for o in range(len(all_ops[j]))
    }

    flow_sum = pulp.LpVariable.dicts("flowtime", jobs, lowBound=0)
    deviation_vars = {}

    # Zielfunktion: Flow + Abweichung
    for j in range(n):
        job = jobs[j]
        seq = all_ops[j]
        d_last = seq[-1][2]
        prob += flow_sum[job] == starts[(j, len(seq)-1)] + d_last - arrival[job]

        for o, (op_id, _, _) in enumerate(seq):
            key = (job, op_id)
            if key in original_start:
                dev = pulp.LpVariable(f"dev_{j}_{o}", lowBound=0)
                deviation_vars[(j, o)] = dev
                prob += dev >= starts[(j, o)] - original_start[key]
                prob += dev >= original_start[key] - starts[(j, o)]

    prob += r * pulp.lpSum(flow_sum.values()) + (1 - r) * pulp.lpSum(deviation_vars.values())

    # 4. Technologische Reihenfolge + Flow-Zuordnung + Startbeschränkung
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        earliest = max(arrival[job], last_executed_end.get(job, reschedule_start))
        prob += starts[(j, 0)] >= earliest
        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev

    # 5. Maschinenkonflikte (inkl. Fixierte)
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

    # 6. Solver auswählen
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

    # 7. Ergebnisse extrahieren
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            ed = st + d
            records.append({
                "Job": job,
                "Operation": op_id,
                "Arrival": arrival[job],
                "Machine": m,
                "Start": round(st, 2),
                "Processing Time": d,
                "Flow time": round(ed - arrival[job], 2),
                "End": round(ed, 2)
            })

    df_schedule = (
        pd.DataFrame(records)
        .sort_values(["Start", "Job", "Operation"])
        .reset_index(drop=True)
    )

    # 8. Logging
    print("\nSolver-Informationen:")
    print(f"  Zielfunktionswert       : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")

    return df_schedule




# Flowtime Rescheduling with Arrivals (simple) ----------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------

def solve_jssp_individual_flowtime_with_fixed_ops(df_jssp: pd.DataFrame, df_arrivals: pd.DataFrame, df_executed: pd.DataFrame,
                                                  reschedule_start: float = 1440.0, solver: str = 'HiGHS', epsilon: float = 0.0,
                                                  sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert die Summe der Flow Times eines Job-Shop-Problems mit fixierten Operationen.

    Bereits ausgeführte Operationen (aus df_executed) bleiben erhalten. Neue Operationen werden ab
    reschedule_start neu geplant. Maschinenkonflikte und technologische Reihenfolge
    werden vollständig berücksichtigt.

    Zielfunktion: sum_j (Endzeit_j - Arrival_j)

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - df_arrivals: DataFrame mit ['Job','Arrival'].
    - df_executed: DataFrame mit ['Job','Machine','Start','End'].
    - solver: 'CBC' oder 'HiGHS' (case-insensitive).
    - epsilon: Pufferzeit zwischen Operationen auf derselben Maschine.
    - sort_ascending: Sortierung der Ankunftszeiten.
    - reschedule_start: Zeitpunkt ab dem neu geplant werden soll.
    - **solver_args: Weitere Solver-Parameter wie msg=True, timeLimit=1200 etc.

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Machine','Start','Processing Time','Flow time','End'].
    """

    # 1. Vorverarbeitung
    df_arrivals = df_arrivals.sort_values("Arrival", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals.set_index("Job")["Arrival"].to_dict()
    jobs = df_arrivals["Job"].tolist()

    # BigM berechnen (Worst Case)
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_arrival = max(arrival.values())
    bigM_raw = max_arrival - min_arrival + sum_proc_time
    bigM = math.ceil(bigM_raw / 1000) * 1000
    print(f"BigM: {bigM}")

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

    # Fixierte Operationen aus df_executed
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    fixed_ops = {
        m: list(grp[["Start", "End", "Job"]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby("Job")["End"].max().to_dict()

    # 3. LP-Modell
    prob = pulp.LpProblem("JSSP_FlowTime_Fixed", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]])
        for j in range(n)
        for o in range(len(all_ops[j]))
    }

    flow_sum = pulp.LpVariable.dicts("flowtime", jobs, lowBound=0)

    # Zielfunktion: Minimierung der Summe aller Flow Times
    prob += pulp.lpSum([flow_sum[job] for job in jobs])

    # 4. Technologische Reihenfolge + Flow-Zuordnung + Startbeschränkung
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        # Startzeit der ersten Operation darf nicht vor letzter ausgeführter enden
        earliest = max(arrival[job], last_executed_end.get(job, reschedule_start))
        prob += starts[(j, 0)] >= earliest
        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev
        d_last = seq[-1][2]
        prob += flow_sum[job] == starts[(j, len(seq) - 1)] + d_last - arrival[job]

    # 5. Maschinenkonflikte (inkl. Fixierte)
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

    # 6. Solver auswählen
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

    # 7. Ergebnisse extrahieren
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            ed = st + d
            records.append({
                "Job": job,
                "Operation": op_id,
                "Arrival": arrival[job],
                "Machine": m,
                "Start": round(st, 2),
                "Processing Time": d,
                "Flow time": round(ed - arrival[job], 2),
                "End": round(ed, 2)
            })

    df_schedule = (
        pd.DataFrame(records)
        .sort_values(["Start", "Job", "Operation"])
        .reset_index(drop=True)
    )

    # 8. Logging
    print("\nSolver-Informationen:")
    print(f"  Summe Flow Times        : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")

    return df_schedule


def solve_jssp_weighted_flowtime_with_fixed_ops(df_jssp: pd.DataFrame, df_arrivals: pd.DataFrame, df_executed: pd.DataFrame,
                                                  reschedule_start: float = 1440.0, solver: str = 'HiGHS', epsilon: float = 0.0,
                                                  sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert die gewichtete Summe der Flow Times eines Job-Shop-Problems mit fixierten Operationen.

    Bereits ausgeführte Operationen (aus df_executed) bleiben erhalten. Neue Operationen werden ab
    reschedule_start neu geplant. Maschinenkonflikte und technologische Reihenfolge
    werden vollständig berücksichtigt.

    Zielfunktion: sum_j Gewicht_j * (Endzeit_j - Arrival_j), wobei Gewicht_j = sqrt(LastArrival / (1 + Arrival_j))

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - df_arrivals: DataFrame mit ['Job','Arrival'].
    - df_executed: DataFrame mit ['Job','Machine','Start','End'].
    - solver: 'CBC' oder 'HiGHS' (case-insensitive).
    - epsilon: Pufferzeit zwischen Operationen auf derselben Maschine.
    - sort_ascending: Sortierung der Ankunftszeiten.
    - reschedule_start: Zeitpunkt ab dem neu geplant werden soll.
    - **solver_args: Weitere Solver-Parameter wie msg=True, timeLimit=1200 etc.

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Machine','Start','Processing Time','Flow time','End'].
    """

    # 1. Vorverarbeitung
    df_arrivals = df_arrivals.sort_values("Arrival", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals.set_index("Job")["Arrival"].to_dict()
    jobs = df_arrivals["Job"].tolist()
    last_arrival = max(arrival.values())
    weights = {job: math.sqrt(last_arrival / (1 + arrival[job])) for job in jobs}

    # BigM berechnen (Worst Case)
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_arrival = max(arrival.values())
    bigM_raw = max_arrival - min_arrival + sum_proc_time
    bigM = math.ceil(bigM_raw / 1000) * 1000
    print(f"BigM: {bigM}")

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

    # Fixierte Operationen aus df_executed
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    fixed_ops = {
        m: list(grp[["Start", "End", "Job"]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby("Job")["End"].max().to_dict()

    # 3. LP-Modell
    prob = pulp.LpProblem("JSSP_WeightedFlowTime_Fixed", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]])
        for j in range(n)
        for o in range(len(all_ops[j]))
    }

    flow_sum = pulp.LpVariable.dicts("flowtime", jobs, lowBound=0)

    # Zielfunktion: Minimierung der gewichteten Flow Times
    prob += pulp.lpSum([weights[job] * flow_sum[job] for job in jobs])

    # 4. Technologische Reihenfolge + Flow-Zuordnung + Startbeschränkung
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        earliest = max(arrival[job], last_executed_end.get(job, reschedule_start))
        prob += starts[(j, 0)] >= earliest
        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev
        d_last = seq[-1][2]
        prob += flow_sum[job] == starts[(j, len(seq) - 1)] + d_last - arrival[job]

    # 5. Maschinenkonflikte (inkl. Fixierte)
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

    # 6. Solver auswählen
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

    # 7. Ergebnisse extrahieren
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            ed = st + d
            records.append({
                "Job": job,
                "Operation": op_id,
                "Arrival": arrival[job],
                "Machine": m,
                "Start": round(st, 2),
                "Processing Time": d,
                "Flow time": round(ed - arrival[job], 2),
                "End": round(ed, 2)
            })

    df_schedule = (
        pd.DataFrame(records)
        .sort_values(["Start", "Job", "Operation"])
        .reset_index(drop=True)
    )

    # 8. Logging
    print("\nSolver-Informationen:")
    print(f"  Gewichtete Flow Time     : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")

    return df_schedule

