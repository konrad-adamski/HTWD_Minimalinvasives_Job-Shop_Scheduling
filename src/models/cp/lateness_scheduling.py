from ortools.sat.python import cp_model
import pandas as pd
import math

# Lateness Scheduling -----------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------

def solve_jssp_sum(df_jssp: pd.DataFrame, df_times: pd.DataFrame, job_column: str = "Job",
                                        schedule_start: float = 0.0, sort_ascending: bool = False,
                                        msg: bool = False, timeLimit: int = 3600, gapRel: float = 0.0) -> pd.DataFrame:

    model = cp_model.CpModel()

    # === Vorbereitung: Zeitdaten und Sortierung ===
    df_times = df_times.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_times.set_index(job_column)["Arrival"].to_dict()
    deadline = df_times.set_index(job_column)["Deadline"].to_dict()
    jobs = df_times[job_column].tolist()

    # === Strukturierung der Operationen ===
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

    # === Planungshorizont bestimmen ===
    horizon = int(df_jssp["Processing Time"].sum() + max(deadline.values()))

    # === Variablen definieren ===
    starts, ends, intervals = {}, {}, {}
    abs_lateness_vars = []

    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            suffix = f"{j}_{o}"
            start = model.NewIntVar(0, horizon, f"start_{suffix}")
            end = model.NewIntVar(0, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, d, end, f"interval_{suffix}")
            starts[(j, o)] = start
            ends[(j, o)] = end
            intervals[(j, o)] = (interval, m)

    # === Nebenbedingungen: Lateness, Startbedingungen, technologische Reihenfolge ===
    for j, job in enumerate(jobs):
        last_op = len(all_ops[j]) - 1
        job_end = ends[(j, last_op)]

        lateness = model.NewIntVar(-horizon, horizon, f"lateness_{j}")
        abs_lateness = model.NewIntVar(0, horizon, f"abs_lateness_{j}")
        model.Add(lateness == job_end - deadline[job])
        model.AddAbsEquality(abs_lateness, lateness)
        abs_lateness_vars.append(abs_lateness)

        model.Add(starts[(j, 0)] >= max(arrival[job], int(math.ceil(schedule_start))))

        for o in range(1, len(all_ops[j])):
            model.Add(starts[(j, o)] >= ends[(j, o - 1)])

    # === Maschinenkonflikte vermeiden ===
    for m in machines:
        machine_intervals = [intervals[(j, o)][0] for (j, o), (iv, mach) in intervals.items() if mach == m]
        model.AddNoOverlap(machine_intervals)

    # === Zielfunktion definieren ===
    model.Minimize(sum(abs_lateness_vars))

    # === Solver-Konfiguration ===
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.max_time_in_seconds = timeLimit
    solver.parameters.relative_gap_limit = gapRel

    status = solver.Solve(model)

    # === Ergebnis extrahieren ===
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        records = get_records_from_cp(
            jobs, all_ops, starts, arrival, deadline, solver,
            job_column=job_column,
            df_times=df_times
        )
        df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", job_column, "Operation"]).reset_index(drop=True)
    else:
        print(f"\nSolver-Status         : {solver.StatusName(status)}")
        print("No feasible solution found!")
        df_schedule = pd.DataFrame()

    # === Logging ===
    print(f"\nSolver-Status           : {solver.StatusName(status)}")
    print(f"Summe Absolute Lateness : {solver.ObjectiveValue()}")
    print(f"Best Objective Bound    : {solver.BestObjectiveBound()}")
    print(f"Laufzeit                : {solver.WallTime():.2f} Sekunden")

    return df_schedule

def solve_jssp_sum_by_tardiness_and_earliness(df_jssp: pd.DataFrame, df_times: pd.DataFrame, job_column: str = "Job",
                                                      w_t: int = 5, w_e: int = 1, schedule_start: float = 0.0,
                                                      sort_ascending: bool = False,
                                                      msg: bool = False, timeLimit: int = 3600, gapRel: float = 0.0) -> pd.DataFrame:
    model = cp_model.CpModel()

    w_t = int(w_t)
    w_e = int(w_e)

    # === Vorbereitung: Arrival/Deadline extrahieren ===
    df_times = df_times.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_times.set_index(job_column)["Arrival"].to_dict()
    deadline = df_times.set_index(job_column)["Deadline"].to_dict()
    jobs = df_times[job_column].tolist()

    # === Operationen je Job strukturieren ===
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

    for j, job in enumerate(jobs):
        last_op_index = len(all_ops[j]) - 1
        job_end = ends[(j, last_op_index)]

        lateness = model.NewIntVar(-horizon, horizon, f"lateness_{j}")
        model.Add(lateness == job_end - deadline[job])

        tardiness = model.NewIntVar(0, horizon, f"tardiness_{j}")
        model.AddMaxEquality(tardiness, [lateness, 0])
        term_tardiness = model.NewIntVar(0, horizon * w_t, f"term_tardiness_{j}")
        model.Add(term_tardiness == w_t * tardiness)
        weighted_terms.append(term_tardiness)

        earliness = model.NewIntVar(0, horizon, f"earliness_{j}")
        model.AddMaxEquality(earliness, [-lateness, 0])
        term_earliness = model.NewIntVar(0, horizon * w_e, f"term_earliness_{j}")
        model.Add(term_earliness == w_e * earliness)
        weighted_terms.append(term_earliness)

        model.Add(starts[(j, 0)] >= max(arrival[job], int(math.ceil(schedule_start))))

        for o in range(1, len(all_ops[j])):
            model.Add(starts[(j, o)] >= ends[(j, o - 1)])

    for m in machines:
        machine_intervals = [intervals[(j, o)][0] for (j, o), (iv, mach) in intervals.items() if mach == m]
        model.AddNoOverlap(machine_intervals)

    model.Minimize(sum(weighted_terms))

    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.relative_gap_limit = gapRel
    solver.parameters.max_time_in_seconds = timeLimit
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        records = get_records_from_cp(jobs, all_ops, starts, arrival, deadline, solver,
                                      job_column=job_column, df_times=df_times)
        df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", job_column, "Operation"]).reset_index(drop=True)
    else:
        print(f"\nSolver-Status         : {solver.StatusName(status)}")
        print("No feasible solution found!")
        df_schedule = pd.DataFrame()

    print(f"\nSolver-Status         : {solver.StatusName(status)}")
    print(f"Objective Value       : {solver.ObjectiveValue():.2f}")
    print(f"Best Objective Bound  : {solver.BestObjectiveBound():.2f}")
    print(f"Laufzeit              : {solver.WallTime():.2f} Sekunden")

    return df_schedule



def get_records_from_cp(jobs, all_ops, starts, arrival, deadline, solver, job_column="Job", df_times=None):
    """
    Erstellt Scheduling-Records mit Tardiness- und Earliness-Berechnung aus CP-Solver-Ergebnissen.

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
    - Liste von Dicts mit allen Scheduling-Informationen pro Operation.
    """

    # 1. Optionales Mapping: Job → Production_Plan_ID
    if df_times is not None and "Routing_ID" in df_times.columns:
        job_routing = df_times.set_index(job_column)["Routing_ID"].to_dict()
    else:
        job_routing = {}

    # 2. Records erzeugen
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = solver.Value(starts[(j, o)])
            ed = st + d
            lateness = ed - deadline[job]

            record = {
                job_column: job,
            }
            if job in job_routing:
                record["Routing_ID"] = job_routing[job]
            record.update({
                "Operation": op_id,
                "Machine": m,
                "Arrival": arrival[job],
                "Deadline": deadline[job],
                "Start": st,
                "Processing Time": d,
                "End": ed,
                "Lateness": lateness,
                "Tardiness": max(0, lateness),
                "Earliness": max(0, -lateness)
            })
            records.append(record)

    return records