from src.models.cp.builder import get_records_from_cp
from ortools.sat.python import cp_model
import pandas as pd
import math


def solve_jssp_sum_with_fixed_ops(
        df_jssp: pd.DataFrame, df_arrivals_deadlines: pd.DataFrame,df_executed: pd.DataFrame,
        reschedule_start: float = 1440.0, job_column: str = "Job",  earliest_start_column: str = "Arrival",
        sort_ascending: bool = False, msg: bool = False, timeLimit: int = 3600,
        gapRel: float = 0.0) -> pd.DataFrame:
    model = cp_model.CpModel()

    # === 1. Vorbereitung: Ankunfts- und Deadline-Daten ===
    df_arrivals_deadlines = df_arrivals_deadlines.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    earliest_start = df_arrivals_deadlines.set_index(job_column)[earliest_start_column].to_dict()
    deadline = df_arrivals_deadlines.set_index(job_column)["Deadline"].to_dict()
    jobs = df_arrivals_deadlines[job_column].tolist()

    # === 2. Operationen gruppieren und Maschinenmenge bestimmen ===
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

    # === 3. Planungshorizont grob schätzen ===
    horizon = int(df_jssp["Processing Time"].sum() + max(deadline.values()))

    # === 4. Feste Operationen auf Maschinen vorbereiten (aus df_executed) ===
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start]
    fixed_ops = {
        m: list(grp[["Start", "End"]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }

    # === 5. Variablen definieren ===
    starts, ends, intervals = {}, {}, {}
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            suffix = f"{j}_{o}"
            start = model.NewIntVar(0, horizon, f"start_{suffix}")
            end = model.NewIntVar(0, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, d, end, f"interval_{suffix}")
            starts[(j, o)] = start
            ends[(j, o)] = end
            intervals[(j, o)] = (interval, m)

    # === 6. Tardiness-Zielfunktion und Nebenbedingungen je Job ===
    tardiness_vars = []
    for j, job in enumerate(jobs):
        last_op = len(all_ops[j]) - 1
        job_end = ends[(j, last_op)]
        tardiness = model.NewIntVar(0, horizon, f"tardiness_{j}")
        model.Add(tardiness >= job_end - deadline[job])
        tardiness_vars.append(tardiness)

        model.Add(starts[(j, 0)] >= earliest_start[job])
        model.Add(starts[(j, 0)] >= int(reschedule_start))
        if job in df_executed[job_column].values:
            last_fixed_end = df_executed[df_executed[job_column] == job]["End"].max()
            model.Add(starts[(j, 0)] >= int(math.ceil(last_fixed_end)))

        for o in range(1, len(all_ops[j])):
            model.Add(starts[(j, o)] >= ends[(j, o - 1)])

    # === 7. Maschinenkonflikte (inkl. fixer Intervalle) ===
    for m in machines:
        machine_intervals = [
            interval for (j, o), (interval, mach) in intervals.items() if mach == m
        ]
        for fixed_start, fixed_end in fixed_ops.get(m, []):
            start = math.floor(fixed_start)
            end = math.ceil(fixed_end)
            duration = end - start
            fixed_interval = model.NewIntervalVar(start, duration, end, f"fixed_{m}_{end}")
            machine_intervals.append(fixed_interval)

        model.AddNoOverlap(machine_intervals)

    # === 8. Zielfunktion: Summe aller Tardiness minimieren ===
    model.Minimize(sum(tardiness_vars))

    # === 9. Solver-Konfiguration ===
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.relative_gap_limit = gapRel
    solver.parameters.max_time_in_seconds = timeLimit
    status = solver.Solve(model)

    # === 10. Ergebnis extrahieren ===
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        records = get_records_from_cp(
            jobs=jobs, all_ops=all_ops, starts=starts,
            solver=solver, job_column=job_column,df_times=df_arrivals_deadlines
        )
        df_schedule = pd.DataFrame.from_records(records).sort_values(by=["Start", job_column, "Operation"]).reset_index(drop=True)
    else:
        print("No solution was found within the time limit!")
        df_schedule = pd.DataFrame()

    print(f"\nSolver-Status         : {solver.StatusName(status)}")
    print(f"Summe Tardiness       : {solver.ObjectiveValue()}")
    print(f"Best Objective Bound  : {solver.BestObjectiveBound()}")
    print(f"Laufzeit              : {solver.WallTime():.2f} Sekunden")

    return df_schedule


def solve_jssp_sum_with_devpen(
    df_jssp: pd.DataFrame,
    df_arrivals_deadlines: pd.DataFrame,
    df_executed: pd.DataFrame,
    df_original_plan: pd.DataFrame,
    reschedule_start: float = 1440.0,
    job_column: str = "Job",
    earliest_start_column: str = "Arrival",
    sort_ascending: bool = False,
    r: float = 0.5,
    msg: bool = False,
    timeLimit: int = 3600,
    gapRel: float = 0.0
) -> pd.DataFrame:

    # === 1. Initialisierung und Gewichtung ===
    model = cp_model.CpModel()
    r_scaled = int(round(r * 100))

    # === 2. Ankunfts- und Deadline-Daten ===
    df_arrivals_deadlines = df_arrivals_deadlines.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    earliest_start = df_arrivals_deadlines.set_index(job_column)[earliest_start_column].to_dict()
    deadline = df_arrivals_deadlines.set_index(job_column)["Deadline"].to_dict()
    jobs = df_arrivals_deadlines[job_column].tolist()

    # === 3. Originale Startzeiten für Deviation ===
    deviation_ops = set(df_jssp[[job_column, "Operation"]].apply(tuple, axis=1)) & \
                    set(df_original_plan[[job_column, "Operation"]].apply(tuple, axis=1))
    original_start = {
        (row[job_column], row["Operation"]): int(round(row["Start"]))
        for _, row in df_original_plan.iterrows()
        if (row[job_column], row["Operation"]) in deviation_ops
    }

    # === 4. Operationen vorbereiten ===
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

    # === 5. Planungshorizont und Fixierungen ===
    horizon = int(df_jssp["Processing Time"].sum() + max(deadline.values()))
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start]
    fixed_ops = {
        m: list(grp[["Start", "End"]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby(job_column)["End"].max().to_dict()

    # === 6. Variablen definieren ===
    starts, ends, intervals = {}, {}, {}
    tardiness_vars, deviation_terms = [], []

    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            suffix = f"{j}_{o}"
            start = model.NewIntVar(0, horizon, f"start_{suffix}")
            end = model.NewIntVar(0, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, d, end, f"interval_{suffix}")
            starts[(j, o)] = start
            ends[(j, o)] = end
            intervals[(j, o)] = (interval, m)

    # === 7. Nebenbedingungen und Zielfunktionsterme ===
    for j, job in enumerate(jobs):
        last_op = len(all_ops[j]) - 1
        job_end = ends[(j, last_op)]

        # Tardiness
        tardiness = model.NewIntVar(0, horizon, f"tardiness_{j}")
        model.Add(tardiness >= job_end - deadline[job])
        tardiness_vars.append(tardiness)

        # Startbedingungen
        model.Add(starts[(j, 0)] >= max(earliest_start[job], int(reschedule_start)))
        if job in last_executed_end:
            model.Add(starts[(j, 0)] >= int(math.ceil(last_executed_end[job])))

        # Technologische Reihenfolge
        for o in range(1, len(all_ops[j])):
            model.Add(starts[(j, o)] >= ends[(j, o - 1)])

        # Deviation-Penalty
        for o, (op_id, _, _) in enumerate(all_ops[j]):
            key = (job, op_id)
            if key in original_start:
                diff = model.NewIntVar(-horizon, horizon, f"diff_{j}_{o}")
                dev = model.NewIntVar(0, horizon, f"dev_{j}_{o}")
                model.Add(diff == starts[(j, o)] - original_start[key])
                model.AddAbsEquality(dev, diff)
                deviation_terms.append(dev)

    # === 8. Maschinenkonflikte inkl. fixer Intervalle ===
    for m in machines:
        machine_intervals = [interval for (j, o), (interval, mach) in intervals.items() if mach == m]
        for fixed_start, fixed_end in fixed_ops.get(m, []):
            start = math.floor(fixed_start)
            end = math.ceil(fixed_end)
            if end > start:
                duration = end - start
                fixed_interval = model.NewIntervalVar(start, duration, end, f"fixed_{m}_{end}")
                machine_intervals.append(fixed_interval)
        model.AddNoOverlap(machine_intervals)

    # === 9. Zielfunktion kombinieren ===
    weighted_tardiness = model.NewIntVar(0, horizon * len(tardiness_vars), "weighted_tardiness")
    total_deviation = model.NewIntVar(0, horizon * len(deviation_terms), "total_deviation")
    total_cost = model.NewIntVar(0, horizon * len(jobs) * 100, "total_cost")

    model.Add(weighted_tardiness == sum(tardiness_vars))
    model.Add(total_deviation == sum(deviation_terms))
    model.Add(total_cost == r_scaled * weighted_tardiness + (100 - r_scaled) * total_deviation)

    model.Minimize(total_cost)

    # === 10. Lösen und Rückgabe ===
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.max_time_in_seconds = timeLimit
    solver.parameters.relative_gap_limit = gapRel
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        records = get_records_from_cp(
            jobs=jobs, all_ops=all_ops, starts=starts,
            solver=solver, job_column=job_column,df_times=df_arrivals_deadlines
        )
        df_schedule = pd.DataFrame.from_records(records).sort_values(by=["Start", job_column, "Operation"]).reset_index(drop=True)
    else:
        print("No solution found!")
        df_schedule = pd.DataFrame()

    print(f"\nSolver-Status         : {solver.StatusName(status)}")
    print(f"Objective Value       : {solver.ObjectiveValue():.2f}")
    print(f"Best Objective Bound  : {solver.BestObjectiveBound():.2f}")
    print(f"Laufzeit              : {solver.WallTime():.2f} Sekunden")
    print(f"Deviation terms       : {len(deviation_terms)}")

    return df_schedule




