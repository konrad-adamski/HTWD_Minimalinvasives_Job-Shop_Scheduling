from ortools.sat.python import cp_model
import pandas as pd

def solve_jssp_sum(df_jssp: pd.DataFrame, df_arrivals_deadlines: pd.DataFrame,
                                 job_column: str = "Job", sort_ascending: bool = False, msg: bool = False,
                                 timeLimit: int = 3600, gapRel: float = 0.0) -> pd.DataFrame:
    model = cp_model.CpModel()

    # 1. Sortierung und Dictionaries
    df_arrivals_deadlines = df_arrivals_deadlines.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals_deadlines.set_index(job_column)["Arrival"].to_dict()
    deadline = df_arrivals_deadlines.set_index(job_column)["Deadline"].to_dict()
    jobs = df_arrivals_deadlines[job_column].tolist()

    # 2. Operationen gruppieren
    ops_grouped = df_jssp.sort_values([job_column, "Operation"]).groupby(job_column)
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

    # 3. Variablen
    starts, ends, intervals = {}, {}, {}
    tardiness_vars = []

    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            suffix = f"{j}_{o}"
            start = model.NewIntVar(0, horizon, f"start_{suffix}")
            end = model.NewIntVar(0, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, d, end, f"interval_{suffix}")
            starts[(j, o)] = start
            ends[(j, o)] = end
            intervals[(j, o)] = (interval, m)

    # 4. Nebenbedingungen
    for j, job in enumerate(jobs):
        last_op = len(all_ops[j]) - 1
        job_end = ends[(j, last_op)]
        tardiness = model.NewIntVar(0, horizon, f"tardiness_{j}")
        model.Add(tardiness >= job_end - deadline[job])
        tardiness_vars.append(tardiness)
        model.Add(starts[(j, 0)] >= arrival[job])
        for o in range(1, len(all_ops[j])):
            model.Add(starts[(j, o)] >= ends[(j, o - 1)])

    for m in machines:
        machine_intervals = [intervals[(j, o)][0] for (j, o), (interval, mach) in intervals.items() if mach == m]
        model.AddNoOverlap(machine_intervals)

    # 5. Zielfunktion
    model.Minimize(sum(tardiness_vars))

    # 6. Solver-Konfiguration
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.max_time_in_seconds = timeLimit
    solver.parameters.relative_gap_limit = gapRel

    # 7. Lösung
    status = solver.Solve(model)

    # 8. Auswertung
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        records = get_records_from_cp(jobs, all_ops, starts, arrival, deadline, solver,
                                      job_column=job_column, df_times=df_arrivals_deadlines)
        df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", job_column, "Operation"]).reset_index(drop=True)
    else:
        print(f"\nSolver-Status         : {solver.StatusName(status)}")
        print("No feasible solution found!")
        df_schedule = pd.DataFrame()

    # 9. Logging
    print(f"\nSolver-Status         : {solver.StatusName(status)}")
    print(f"Summe Tardiness        : {solver.ObjectiveValue()}")
    print(f"Best Objective Bound   : {solver.BestObjectiveBound()}")
    print(f"Laufzeit               : {solver.WallTime():.2f} Sekunden")

    return df_schedule



def solve_jssp_max(df_jssp: pd.DataFrame, df_arrivals_deadlines: pd.DataFrame,
                                 job_column: str = "Job", sort_ascending: bool = False, msg: bool = False,
                                 timeLimit: int = 3600, gapRel: float = 0.0,) -> pd.DataFrame:
    model = cp_model.CpModel()

    # 1. Vorbereitung der Daten
    df_arrivals_deadlines = df_arrivals_deadlines.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals_deadlines.set_index(job_column)["Arrival"].to_dict()
    deadline = df_arrivals_deadlines.set_index(job_column)["Deadline"].to_dict()
    jobs = df_arrivals_deadlines[job_column].tolist()

    # 2. Operationen gruppieren
    ops_grouped = df_jssp.sort_values([job_column, "Operation"]).groupby(job_column)
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

    # 3. Variablen definieren
    starts, ends, intervals = {}, {}, {}
    tardiness_vars = []
    max_tardiness = model.NewIntVar(0, horizon, "max_tardiness")

    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            suffix = f"{j}_{o}"
            start = model.NewIntVar(0, horizon, f"start_{suffix}")
            end = model.NewIntVar(0, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, d, end, f"interval_{suffix}")
            starts[(j, o)] = start
            ends[(j, o)] = end
            intervals[(j, o)] = (interval, m)

    # 4. Nebenbedingungen pro Job
    for j, job in enumerate(jobs):
        last_op = len(all_ops[j]) - 1
        job_end = ends[(j, last_op)]
        tardiness = model.NewIntVar(0, horizon, f"tardiness_{j}")
        model.Add(tardiness >= job_end - deadline[job])
        model.Add(max_tardiness >= tardiness)
        tardiness_vars.append(tardiness)

        model.Add(starts[(j, 0)] >= arrival[job])
        for o in range(1, len(all_ops[j])):
            model.Add(starts[(j, o)] >= ends[(j, o - 1)])

    # 5. Maschinenrestriktionen
    for m in machines:
        machine_intervals = [intervals[(j, o)][0] for (j, o), (interval, mach) in intervals.items() if mach == m]
        model.AddNoOverlap(machine_intervals)

    # 6. Zielfunktion
    model.Minimize(max_tardiness)

    # 7. Solver konfigurieren
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.relative_gap_limit = gapRel
    solver.parameters.max_time_in_seconds = timeLimit

    # 8. Modell lösen
    status = solver.Solve(model)

    # 9. Ergebnisse extrahieren
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        records = get_records_from_cp(jobs, all_ops, starts, arrival, deadline, solver,
                                      job_column=job_column, df_times=df_arrivals_deadlines)
        df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", job_column, "Operation"]).reset_index(drop=True)
    else:
        print(f"\nSolver-Status         : {solver.StatusName(status)}")
        print("No feasible solution found!")
        df_schedule = pd.DataFrame()

    # 10. Logging
    print(f"\nSolver-Status         : {solver.StatusName(status)}")
    print(f"Maximale Tardiness     : {solver.ObjectiveValue()}")
    print(f"Best Objective Bound   : {solver.BestObjectiveBound()}")
    print(f"Laufzeit               : {solver.WallTime():.2f} Sekunden")

    return df_schedule



def get_records_from_cp(jobs, all_ops, starts, arrival, deadline, solver, job_column="Job", df_times=None):
    """
    Erstellt Scheduling-Records mit Tardiness-Berechnung aus CP-Solver-Ergebnissen.

    Parameter:
    - jobs: Liste der Job-IDs.
    - all_ops: Liste der Operationen je Job [(op_id, machine, duration), ...].
    - starts: Dict mit CP-Startvariablen (solver.Value(var)).
    - arrival: Dict {Job → Arrival}.
    - deadline: Dict {Job → Deadline}.
    - solver: cp_model.CpSolver().
    - job_column: Name der Jobspalte im DataFrame.
    - df_times: Optionaler DataFrame mit 'Production_Plan_ID' je Job.

    Rückgabe:
    - Liste von Dicts mit ['Job','Operation','Arrival','Deadline','Machine',
      'Start','Processing Time','End','Lateness','Tardiness'].
    """

    # 1. Optionales Mapping: Job → Production_Plan_ID
    if df_times is not None and "Production_Plan_ID" in df_times.columns:
        job_production_plan = df_times.set_index(job_column)["Production_Plan_ID"].to_dict()
    else:
        job_production_plan = {}

    # 2. Record-Erstellung
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = solver.Value(starts[(j, o)])
            ed = st + d
            lateness = ed - deadline[job]

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
                "Start": st,
                "Processing Time": d,
                "End": ed,
                "Lateness": lateness,
                "Tardiness": max(0, lateness)
            })
            records.append(record)

    return records
