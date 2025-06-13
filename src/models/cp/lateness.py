import pandas as pd
from ortools.sat.python import cp_model


# Lateness Scheduling -----------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------

def solve_cp_jssp_sum_absolute_lateness(
    df_jssp: pd.DataFrame,
    df_arrivals_deadlines: pd.DataFrame,
    schedule_start: float = 0.0,
    sort_ascending: bool = False,
    msg: bool = False,
    timeLimit: int = 3600,
    gapRel: float = 0.0
) -> pd.DataFrame:
    from ortools.sat.python import cp_model
    import pandas as pd
    import math

    model = cp_model.CpModel()

    # Sortiere nach Deadline, falls gewünscht
    df_arrivals_deadlines = df_arrivals_deadlines.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals_deadlines.set_index("Job")["Arrival"].to_dict()
    deadline = df_arrivals_deadlines.set_index("Job")["Deadline"].to_dict()
    jobs = df_arrivals_deadlines["Job"].tolist()

    # Gruppiere Operationen je Job
    ops_grouped = df_jssp.sort_values(["Job", "Operation"]).groupby("Job")
    all_ops, machines = [], set()
    for job in jobs:
        seq = []
        for _, row in ops_grouped.get_group(job).iterrows():
            op_id = int(row["Operation"])
            m = str(row["Machine"])
            d = int(round(row["Processing Time"]))
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    horizon = int(df_jssp["Processing Time"].sum() + max(deadline.values()))

    # Variablen definieren
    starts, ends, intervals = {}, {}, {}
    abs_lateness_vars = []

    for j, job in enumerate(jobs):
        seq = all_ops[j]
        for o, (op_id, m, d) in enumerate(seq):
            suffix = f"{j}_{o}"
            start = model.NewIntVar(0, horizon, f"start_{suffix}")
            end = model.NewIntVar(0, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, d, end, f"interval_{suffix}")
            starts[(j, o)] = start
            ends[(j, o)] = end
            intervals[(j, o)] = (interval, m)

    # Lateness-Berechnung und Nebenbedingungen pro Job
    for j, job in enumerate(jobs):
        last_op = len(all_ops[j]) - 1
        job_end = ends[(j, last_op)]

        # Lateness kann positiv (zu spät) oder negativ (zu früh) sein
        lateness = model.NewIntVar(-horizon, horizon, f"lateness_{j}")
        abs_lateness = model.NewIntVar(0, horizon, f"abs_lateness_{j}")
        model.Add(lateness == job_end - deadline[job])
        model.AddAbsEquality(abs_lateness, lateness)
        abs_lateness_vars.append(abs_lateness)

        # Startzeitbedingung (max von Arrival und schedule_start)
        model.Add(starts[(j, 0)] >= max(arrival[job], int(math.ceil(schedule_start))))

        # Technologische Reihenfolge
        for o in range(1, len(all_ops[j])):
            model.Add(starts[(j, o)] >= ends[(j, o - 1)])

    # Maschinenrestriktionen
    for m in machines:
        machine_intervals = [intervals[(j, o)][0] for (j, o), (interval, mach) in intervals.items() if mach == m]
        model.AddNoOverlap(machine_intervals)

    # Zielfunktion
    model.Minimize(sum(abs_lateness_vars))

    # Solver starten
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.max_time_in_seconds = timeLimit
    solver.parameters.relative_gap_limit = gapRel

    status = solver.Solve(model)

    # Lösung extrahieren (nur bei OPTIMAL oder FEASIBLE)
    records = []
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        for j, job in enumerate(jobs):
            for o, (op_id, m, d) in enumerate(all_ops[j]):
                st = solver.Value(starts[(j, o)])
                ed = st + d
                lateness = ed - deadline[job]
                records.append({
                    "Job": job,
                    "Operation": op_id,
                    "Machine": m,
                    "Arrival": arrival[job],
                    "Deadline": deadline[job],
                    "Start": st,
                    "Processing Time": d,
                    "End": ed,
                    "Lateness": lateness,
                    "Absolute Lateness": abs(lateness)
                })

        df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", "Job", "Operation"]).reset_index(drop=True)
    else:
        print(f"\nSolver-Status         : {solver.StatusName(status)}")
        print("No feasible solution found!")
        df_schedule = pd.DataFrame()

    # Logging
    print(f"\nSolver-Status         : {solver.StatusName(status)}")
    print(f"Summe Absolute Lateness : {solver.ObjectiveValue()}")
    print(f"Best Objective Bound    : {solver.BestObjectiveBound()}")
    print(f"Laufzeit                : {solver.WallTime():.2f} Sekunden")

    return df_schedule

def solve_cp_jssp_lateness_by_tardiness_and_earliness(
    df_jssp: pd.DataFrame,
    df_arrivals_deadlines: pd.DataFrame,
    w_t: int = 5,
    w_e: int = 1,
    schedule_start: float = 0.0,
    sort_ascending: bool = False,
    msg: bool = False,
    timeLimit: int = 3600,
    gapRel: float = 0.0
) -> pd.DataFrame:
    """
    Solves a Job-Shop Scheduling Problem (JSSP) using Constraint Programming with:
    - weighted tardiness (late completion),
    - weighted earliness (early completion),
    - optional global scheduling start time (schedule_start).

    Parameters:
        df_jssp: DataFrame with job-shop structure: ['Job','Operation','Machine','Processing Time']
        df_arrivals_deadlines: DataFrame with ['Job','Arrival','Deadline'] for each job
        w_t: Weight for tardiness penalty (default: 5)
        w_e: Weight for earliness penalty (default: 1)
        schedule_start: Earliest time from which operations may be scheduled (default: 0.0)
        sort_ascending: If True, jobs are sorted by deadline ascending (default: False)
        msg: Verbose solver output (default: False)
        timeLimit: Maximum solver time in seconds (default: 3600)
        gapRel: Acceptable relative gap for feasible solutions (default: 0.0)

    Returns:
        df_schedule: DataFrame with planned operations, start/end times, and lateness metrics.
    """
    from ortools.sat.python import cp_model
    import pandas as pd
    import math

    model = cp_model.CpModel()

    # Gewichte als ganze Zahlen für CP-Modell
    w_t = int(w_t)
    w_e = int(w_e)

    # === Vorbereitung: Jobliste, Ankunft, Deadlines ===
    df_arrivals_deadlines = df_arrivals_deadlines.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals_deadlines.set_index("Job")["Arrival"].to_dict()
    deadline = df_arrivals_deadlines.set_index("Job")["Deadline"].to_dict()
    jobs = df_arrivals_deadlines["Job"].tolist()

    # === Operationen je Job strukturieren und Maschinen erfassen ===
    ops_grouped = df_jssp.sort_values(["Job", "Operation"]).groupby("Job")
    all_ops, machines = [], set()
    for job in jobs:
        seq = []
        for _, row in ops_grouped.get_group(job).iterrows():
            op_id = int(row["Operation"])
            m = str(row["Machine"])
            d = int(round(row["Processing Time"]))
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    # === Grobe obere Schranke für Planungshorizont ===
    horizon = int(df_jssp["Processing Time"].sum() + max(deadline.values()))

    # === Zeitvariablen und Intervalle anlegen ===
    starts, ends, intervals = {}, {}, {}
    weighted_terms = []

    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            suffix = f"{j}_{o}"
            start = model.NewIntVar(0, horizon, f"start_{suffix}")
            end = model.NewIntVar(0, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, d, end, f"interval_{suffix}")
            starts[(j, o)] = start
            ends[(j, o)] = end
            intervals[(j, o)] = (interval, m)

    # === Zielgrößen pro Job berechnen ===
    for j, job in enumerate(jobs):
        last_op_index = len(all_ops[j]) - 1
        job_end = ends[(j, last_op_index)]

        # Lateness = tatsächliches Ende - Deadline
        lateness = model.NewIntVar(-horizon, horizon, f"lateness_{j}")
        model.Add(lateness == job_end - deadline[job])

        # Tardiness = max(0, Lateness)
        tardiness = model.NewIntVar(0, horizon, f"tardiness_{j}")
        model.AddMaxEquality(tardiness, [lateness, 0])
        term_tardiness = model.NewIntVar(0, horizon * w_t, f"term_tardiness_{j}")
        model.Add(term_tardiness == w_t * tardiness)
        weighted_terms.append(term_tardiness)

        # Earliness = max(0, -Lateness)
        earliness = model.NewIntVar(0, horizon, f"earliness_{j}")
        model.AddMaxEquality(earliness, [-lateness, 0])
        term_earliness = model.NewIntVar(0, horizon * w_e, f"term_earliness_{j}")
        model.Add(term_earliness == w_e * earliness)
        weighted_terms.append(term_earliness)

        # Startbedingung für erste Operation: max(Arrival, schedule_start)
        # Verhindert Einplanung vor Beginn des neuen Planungshorizonts
        model.Add(starts[(j, 0)] >= max(arrival[job], int(math.ceil(schedule_start))))

        # Technologische Reihenfolge: O_i+1 nach O_i
        for o in range(1, len(all_ops[j])):
            model.Add(starts[(j, o)] >= ends[(j, o - 1)])

    # === Maschinenrestriktionen: keine Überlappung pro Maschine ===
    for m in machines:
        machine_intervals = [intervals[(j, o)][0] for (j, o), (iv, mach) in intervals.items() if mach == m]
        model.AddNoOverlap(machine_intervals)

    # === Zielfunktion: gewichtete Summe aus Tardiness und Earliness minimieren ===
    model.Minimize(sum(weighted_terms))

    # === Solver konfigurieren ===
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.relative_gap_limit = gapRel
    solver.parameters.max_time_in_seconds = timeLimit
    status = solver.Solve(model)

    # === Ergebnis extrahieren ===
    records = []
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        for j, job in enumerate(jobs):
            for o, (op_id, m, d) in enumerate(all_ops[j]):
                st = solver.Value(starts[(j, o)])
                ed = st + d
                lateness_val = ed - deadline[job]
                records.append({
                    "Job": job,
                    "Operation": op_id,
                    "Machine": m,
                    "Arrival": arrival[job],
                    "Deadline": deadline[job],
                    "Start": st,
                    "Processing Time": d,
                    "End": ed,
                    "Lateness": lateness_val,
                    "Tardiness": max(0, lateness_val),
                    "Earliness": max(0, -lateness_val)
                })

        df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", "Job", "Operation"]).reset_index(drop=True)
    else:
        print(f"\nSolver-Status         : {solver.StatusName(status)}")
        print("No feasible solution found!")
        df_schedule = pd.DataFrame()

    # === Zusammenfassung der Lösung ===
    print(f"\nSolver-Status         : {solver.StatusName(status)}")
    print(f"Objective Value       : {solver.ObjectiveValue():.2f}")
    print(f"Best Objective Bound  : {solver.BestObjectiveBound():.2f}")
    print(f"Laufzeit              : {solver.WallTime():.2f} Sekunden")

    return df_schedule
