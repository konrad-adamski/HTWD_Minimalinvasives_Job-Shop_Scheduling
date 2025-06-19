import math
import pandas as pd
import pulp
import time


def solve_jssp(df_jssp: pd.DataFrame, job_column: str = 'Job', 
                        solver: str = 'CBC', epsilon: float = 0.0, var_cat: str = 'Continuous',
                        time_limit: int | None = 10800, **solver_args):
    """
    Minimiert den Makespan eines Job-Shop-Problems auf Basis eines DataFrames,
    mit dynamisch berechnetem Big-M statt statischem 1e5.

    Parameter:
    - df_jssp: DataFrame mit Spalten [job_column, 'Operation','Machine','Processing Time'].
    - job_column: Name der Spalte, die den Produktionsauftrag (Job) bezeichnet.
    - solver: 'CBC' oder 'HiGHS' (case-insensitive).
    - epsilon: Pufferzeit in Minuten zwischen zwei Jobs auf derselben Maschine.
    - time_limit: Zeitlimit für den Solver in Sekunden (Standard: 10800 s - 3 h).
    - **solver_args: Weitere Keyword-Argumente für den gewählten Solver.

    Rückgabe:
    - df_schedule: DataFrame mit Spalten
      [job_column, 'Operation','Machine','Start','Processing Time','End']
    - makespan_value: minimaler Makespan (float)
    """
    starting_time = time.time()
    # 1) Index
    df = df_jssp.reset_index(drop=False).rename(columns={'index': 'Idx'}).copy()

    # Big-M berechnen
    sum_proc_time = df['Processing Time'].sum()
    bigM = math.ceil(sum_proc_time / 100) * 100
    print(f"BigM: {bigM}")

    # 2) LP-Modell
    prob = pulp.LpProblem('JSSP', pulp.LpMinimize)
    starts = {
        idx: pulp.LpVariable(f'start_{idx}', lowBound=0, cat=var_cat)
        for idx in df['Idx']
    }
    makespan = pulp.LpVariable('makespan', lowBound=0, cat=var_cat)
    prob += makespan

    # 3) Reihenfolge je Job (technologische Abhängigkeiten)
    for job, group in df.groupby(job_column, sort=False):
        seq = group.sort_values('Operation')
        for i in range(len(seq) - 1):
            prev = seq.iloc[i]['Idx']
            curr = seq.iloc[i + 1]['Idx']
            dur_prev = df.loc[df['Idx'] == prev, 'Processing Time'].iat[0]
            prob += starts[curr] >= starts[prev] + dur_prev

    # 4) Maschinenkonflikte mit Big-M
    for _, group in df.groupby('Machine', sort=False):
        ids = group['Idx'].tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                i_idx, j_idx = ids[i], ids[j]
                if df.loc[df['Idx'] == i_idx, job_column].iat[0] == df.loc[df['Idx'] == j_idx, job_column].iat[0]:
                    continue
                di = df.loc[df['Idx'] == i_idx, 'Processing Time'].iat[0]
                dj = df.loc[df['Idx'] == j_idx, 'Processing Time'].iat[0]
                y = pulp.LpVariable(f'y_{i_idx}_{j_idx}', cat='Binary')
                prob += starts[i_idx] + di + epsilon <= starts[j_idx] + bigM * (1 - y)
                prob += starts[j_idx] + dj + epsilon <= starts[i_idx] + bigM * y

    # 5) Makespan-Constraints
    for _, row in df.iterrows():
        idx = int(row['Idx'])
        prob += starts[idx] + row['Processing Time'] <= makespan

    # 6) Solver vorbereiten
    if time_limit is not None:
        solver_args.setdefault('timeLimit', time_limit)
        
    solver_args.setdefault('msg', True)

    solver = solver.upper()
    if solver in ['CBC', 'BRANCH AND CUT']:
        cmd = pulp.PULP_CBC_CMD(**solver_args)
    elif solver == 'HIGHS':
        cmd = pulp.HiGHS_CMD(**solver_args)
    else:
        raise ValueError("Solver must be 'CBC' or 'HiGHS'")

    prob.solve(cmd)

    solving_duration = time.time() - starting_time
    makespan_value = pulp.value(prob.objective)

    # 7) Ergebnis aufbereiten
    df['Start'] = df['Idx'].map(lambda i: round(starts[i].varValue, 2))
    df['End'] = df['Start'] + df['Processing Time']
    df_schedule = df[[job_column, 'Operation', 'Machine', 'Start', 'Processing Time', 'End']].sort_values(['Start', job_column, 'Operation']).reset_index(drop=True)

    # 8) Logging
    print("\nSolver-Informationen:")
    print(f"  Zielfunktionswert       : {round(makespan_value, 2)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")
    print(f"  Laufzeit                : ~{solving_duration:.2f} Sekunden")

    return df_schedule


def solve_jssp_with_arrival(df_jssp: pd.DataFrame, df_arrivals: pd.DataFrame,
                                     job_column: str = 'Job',
                                     solver: str = 'CBC', epsilon: float = 0.0,
                                     var_cat: str = 'Continuous',
                                     time_limit: int | None = 10800, **solver_args) -> tuple[pd.DataFrame, float]:
    """
    Minimiert den Makespan eines Job-Shop-Problems mit Ankunftszeiten.

    Parameter:
    - df_jssp: DataFrame mit [job_column,'Operation','Machine','Processing Time'].
    - df_arrivals: DataFrame mit [job_column,'Arrival'].
    - job_column: Name der Spalte, die den Produktionsauftrag (Job) bezeichnet.
    - solver: 'CBC' oder 'HiGHS' (case-insensitive).
    - epsilon: Pufferzeit in Minuten zwischen zwei Jobs auf derselben Maschine.
    - var_cat: 'Continuous' oder 'Integer' für Startzeit-Variablen.
    - time_limit: Max. Zeit in Sekunden für den Solver.
    - **solver_args: Weitere Solver-Parameter wie msg=True, threads=4 etc.

    Rückgabe:
    - df_schedule: DataFrame mit [job_column,'Operation','Arrival','Machine',
      'Start','Processing Time','Flow time','End'].
    - makespan_value: minimaler Makespan (float)
    """
    start_time = time.time()

    # 1. Vorverarbeitung
    df_arrivals = df_arrivals.reset_index(drop=True)
    arrival = df_arrivals.set_index(job_column)["Arrival"].to_dict()
    jobs = df_arrivals[job_column].tolist()

    # 2. BigM berechnen
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_arrival = max(arrival.values())
    bigM_raw = max_arrival - min_arrival + sum_proc_time
    bigM = math.ceil(bigM_raw / 100) * 100
    print(f"BigM: {bigM}")

    # 3. Operationen gruppieren
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

    # 4. LP-Modell
    prob = pulp.LpProblem("JSSP_Makespan_Arrival", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]], cat=var_cat)
        for j in range(n)
        for o in range(len(all_ops[j]))
    }

    makespan = pulp.LpVariable("makespan", lowBound=0, cat=var_cat)
    prob += makespan

    # 5. Technologische Abhängigkeiten
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev
        d_last = seq[-1][2]
        prob += makespan >= starts[(j, len(seq) - 1)] + d_last

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

    # 7. Solver vorbereiten
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
    makespan_value = pulp.value(prob.objective)

    # 8. Ergebnisse extrahieren
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            ed = st + d
            records.append({
                job_column: job,
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
        .sort_values(["Start", job_column, "Operation"])
        .reset_index(drop=True)
    )

    # 9. Logging
    solving_duration = time.time() - start_time
    print("\nSolver-Informationen:")
    print(f"  Makespan                : {round(makespan_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")
    print(f"  Laufzeit                : ~{solving_duration:.2f} Sekunden")

    return df_schedule