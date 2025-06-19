from ortools.sat.python import cp_model
# pip install ortools==9.8.3296# pip install ortools==9.8.3296
import pandas as pd

def solve_jssp(df_jssp: pd.DataFrame, job_column = 'Job', 
               time_limit: int | None = 10800, msg: bool = False, gapRel: float | None = None) -> pd.DataFrame:
    """
    Minimiert den Makespan eines klassischen Job-Shop-Problems (JSSP) mit einem CP-Modell.

    Parameter:
    - df_jssp: DataFrame mit Spalten [job_column, 'Operation', 'Machine', 'Processing Time'].
    - time_limit: Zeitlimit in Sekunden (Standard: 10800 s - 3 h).
    - msg: Ausgabe des CP-Solvers aktivieren.
    - gapRel: Relatives Gap zur frühzeitigen Beendigung.

    Rückgabe:
    - df_schedule: Zeitplan mit Start- und Endzeiten sowie Maschinenbelegung.
    """
    model = cp_model.CpModel()

    # Vorbereitung
    jobs = sorted(df_jssp[job_column].unique())
    ops_grouped = df_jssp.sort_values([job_column, 'Operation']).groupby(job_column)
    all_ops = []
    machines = set()

    for job in jobs:
        seq = []
        for _, row in ops_grouped.get_group(job).iterrows():
            op_id = int(row["Operation"])
            m = str(row["Machine"])
            d = int(round(row["Processing Time"]))
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    horizon = int(df_jssp["Processing Time"].sum())

    # Variablen
    starts, ends, intervals = {}, {}, {}
    for i, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[i]):
            suffix = f"{i}_{o}"
            start = model.NewIntVar(0, horizon, f"start_{suffix}")
            end = model.NewIntVar(0, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, d, end, f"interval_{suffix}")
            starts[(i, o)] = start
            ends[(i, o)] = end
            intervals[(i, o)] = (interval, m)

    # Makespan-Variable
    makespan = model.NewIntVar(0, horizon, "makespan")
    for i, job in enumerate(jobs):
        last_op = len(all_ops[i]) - 1
        model.Add(ends[(i, last_op)] <= makespan)

    # Technologische Abfolge
    for i in range(len(jobs)):
        for o in range(1, len(all_ops[i])):
            model.Add(starts[(i, o)] >= ends[(i, o - 1)])

    # Maschinenrestriktionen
    for m in machines:
        machine_intervals = [intervals[(i, o)][0] for (i, o), (interval, mach) in intervals.items() if mach == m]
        model.AddNoOverlap(machine_intervals)

    # Zielfunktion
    model.Minimize(makespan)

    # Solver starten
    solver = cp_model.CpSolver()
    if time_limit is not None:
        solver.parameters.max_time_in_seconds = time_limit
        
    solver.parameters.log_search_progress = msg
    
    if gapRel is not None:
        solver.parameters.relative_gap_limit = gapRel

    status = solver.Solve(model)

    # Ergebnis extrahieren
    records = []
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        for i, job in enumerate(jobs):
            for o, (op_id, m, d) in enumerate(all_ops[i]):
                st = solver.Value(starts[(i, o)])
                ed = st + d
                records.append({
                    job_column: job,
                    "Operation": op_id,
                    "Machine": m,
                    "Start": st,
                    "Processing Time": d,
                    "End": ed
                })
        df_schedule = pd.DataFrame(records).sort_values([job_column, "Operation"]).reset_index(drop=True)
    else:
        print(f"\nSolver-Status     : {solver.StatusName(status)}")
        print("Keine zulässige Lösung gefunden.")
        df_schedule = pd.DataFrame()

    # Logging
    print("\nSolver-Informationen:")
    print(f"  Solver-Status    : {solver.StatusName(status)}")
    print(f"  Makespan         : {solver.ObjectiveValue()}")
    print(f"  Best Bound       : {solver.BestObjectiveBound()}")
    print(f"  Laufzeit         : {solver.WallTime():.2f} Sekunden")

    return df_schedule
