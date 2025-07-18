import pandas as pd
import pulp
import math
import time


def solve_jssp(df_jssp: pd.DataFrame, df_arrivals: pd.DataFrame,
                        job_column: str = 'Job',
                        solver: str = 'HiGHS', epsilon: float = 0.0,
                        var_cat: str = 'Continuous', earliest_start_column: str = "Arrival",
                        time_limit: int | None = 10800,
                        sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert die gesamte Flow Time eines Job-Shop-Problems mit Ankunftszeiten.

    Parameter:
    - df_jssp: DataFrame mit [job_column,'Operation','Machine','Processing Time'].
    - df_arrivals: DataFrame mit [job_column,'Arrival'].
    - job_column: Name der Spalte, die den Job identifiziert.
    - solver: 'CBC' oder 'HiGHS' (case-insensitive).
    - epsilon: Pufferzeit in Minuten zwischen zwei Jobs auf derselben Maschine.
    - var_cat: 'Continuous' oder 'Integer' für Startzeit-Variablen.
    - time_limit: Max. Zeit in Sekunden für den Solver.
    - sort_ascending: Sortierung der Jobs nach Ankunft (True = früh zuerst).
    - **solver_args: Weitere Solver-Parameter wie msg=True, threads=4 etc.

    Rückgabe:
    - df_schedule: DataFrame mit [job_column,'Operation','Arrival','Machine',
      'Start','Processing Time','Flow time','End'].
    """
    start_time = time.time()

    # 1) Vorverarbeitung
    df_arrivals = df_arrivals.sort_values(earliest_start_column, ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals.set_index(job_column)[earliest_start_column].to_dict()
    jobs = df_arrivals[job_column].tolist()

    # Big-M berechnen
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_arrival = max(arrival.values())
    bigM_raw = max_arrival - min_arrival + sum_proc_time
    bigM = math.ceil(bigM_raw / 1000) * 1000
    print(f"BigM: {bigM}")

    # 2) Operationen je Job
    ops_grouped = df_jssp.sort_values([job_column, "Operation"]).groupby(job_column)
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

    # 3) LP-Modell
    prob = pulp.LpProblem("JSSP_FlowTime_Arrival", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]], cat=var_cat)
        for j in range(n)
        for o in range(len(all_ops[j]))
    }

    flow_sum = {
        job: pulp.LpVariable(f"flowtime_{job}", lowBound=0, cat=var_cat)
        for job in jobs
    }

    prob += pulp.lpSum(flow_sum.values())

    # 4) Technologische Abhängigkeiten + Flowdefinition
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev

        # Letzte Operation bestimmt die Endzeit → FlowTime = End - Arrival
        d_last = seq[-1][2]
        prob += flow_sum[job] == starts[(j, len(seq) - 1)] + d_last - arrival[job]

    # 5) Maschinenkonflikte mit Big-M
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

    # 6) Solver vorbereiten
    if time_limit is not None:
        solver_args.setdefault("timeLimit", time_limit)
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

    # 7) Ergebnisse extrahieren
    df_schedule = get_schedule_df(jobs, all_ops, starts, df_jssp= df_jssp, df_times = df_arrivals, job_column="Job")

    # 8) Logging
    solving_duration = time.time() - start_time
    print("\nSolver-Informationen:")
    print(f"  Summe Flow Times        : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")
    print(f"  Laufzeit                : ~{solving_duration:.2f} Sekunden")

    return df_schedule



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
    jssp_legacy_cols = [col for col in ["Start", "End", "Deadline"] if col in df_jssp.columns]
    times_legacy_cols = ["End"] if "End" in df_times.columns else []

    jssp_cols_to_remove = jssp_shared_cols_to_drop + jssp_legacy_cols
    df_jssp_filtered = df_jssp.drop(columns=jssp_cols_to_remove, errors="ignore")

    df_times_filtered = df_times.drop(columns=times_legacy_cols, errors="ignore")

    # 2c. Merge aus Originaldaten und Zeitinformationen
    df_info = df_jssp_filtered.merge(df_times_filtered, on=job_column, how="left")

    # 3. Ergebnis
    df_schedule = df_info.merge(df_records, on=[job_column, "Operation", "Machine"], how="left")
    if "Routing_ID" in df_schedule.columns:
        df_schedule = df_schedule[
            [job_column, "Routing_ID"] + [col for col in df_schedule.columns if col not in (job_column, "Routing_ID")]]

    return df_schedule