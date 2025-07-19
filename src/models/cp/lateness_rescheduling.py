from src.models.cp.builder import get_records_from_cp
from ortools.sat.python import cp_model
import pandas as pd
import math

def solve_jssp_advanced(
        df_jssp: pd.DataFrame, df_times: pd.DataFrame, df_executed: pd.DataFrame,
        df_original_plan: pd.DataFrame, job_column: str = "Job", earliest_start_column: str = "Arrival",
        w_t: int = 5, w_e: int = 1, r: float = 0.5, alpha: float = 1.0,
        reschedule_start: float = 1440.0, sort_ascending: bool = False, msg: bool = False,
        timeLimit: int = 3600, gapRel: float = 0.0) -> pd.DataFrame:

    # 1. === Modell erstellen ===
    model = cp_model.CpModel()
    w_t = int(w_t)
    w_e = int(w_e)
    r_scaled = int(round(r * 100))
    alpha_scaled = int(round(alpha * 100))

    # 2. === Vorverarbeitung: Ankunft, Deadline, Jobs ===
    df_times = df_times.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    earliest_start = df_times.set_index(job_column)[earliest_start_column].to_dict()
    deadline = df_times.set_index(job_column)["Deadline"].to_dict()
    jobs = df_times[job_column].tolist()

    # 3. === Schnittmenge von Operationen für Deviation bestimmen ===
    deviation_ops = set(df_jssp[[job_column, "Operation"]].apply(tuple, axis=1)) & \
                    set(df_original_plan[[job_column, "Operation"]].apply(tuple, axis=1))
    original_start = {
        (row[job_column], row["Operation"]): int(round(row["Start"]))
        for _, row in df_original_plan.iterrows()
        if (row[job_column], row["Operation"]) in deviation_ops
    }

    # 4. === Operationen gruppieren, Maschinen erfassen ===
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

    # 5. === Planungshorizont schätzen ===
    horizon = int(df_jssp["Processing Time"].sum() + max(deadline.values()))

    # 6. === Fixierte Operationen vorbereiten ===
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start]
    fixed_ops = {
        m: list(grp[["Start", "End"]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby(job_column)["End"].max().to_dict()

    # 7. === Variablen definieren ===
    starts, ends, intervals = {}, {}, {}
    weighted_terms = []
    deviation_terms = []

    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            suffix = f"{j}_{o}"
            start = model.NewIntVar(0, horizon, f"start_{suffix}")
            end = model.NewIntVar(0, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, d, end, f"interval_{suffix}")
            starts[(j, o)] = start
            ends[(j, o)] = end
            intervals[(j, o)] = (interval, m)

    # 8. === Nebenbedingungen und Zielterme ===
    for j, job in enumerate(jobs):
        last_op = len(all_ops[j]) - 1
        job_end = ends[(j, last_op)]

        # Lateness, Tardiness, Earliness
        lateness = model.NewIntVar(-horizon, horizon, f"lateness_{j}")
        model.Add(lateness == job_end - deadline[job])

        tardiness = model.NewIntVar(0, horizon, f"tardiness_{j}")
        model.AddMaxEquality(tardiness, [lateness, 0])
        term_t = model.NewIntVar(0, horizon * w_t, f"term_t_{j}")
        model.Add(term_t == w_t * tardiness)
        weighted_terms.append(term_t)

        earliness = model.NewIntVar(0, horizon, f"earliness_{j}")
        model.AddMaxEquality(earliness, [-lateness, 0])
        term_e = model.NewIntVar(0, horizon * w_e, f"term_e_{j}")
        model.Add(term_e == w_e * earliness)
        weighted_terms.append(term_e)

        # Startzeitbedingungen
        model.Add(starts[(j, 0)] >= max(earliest_start[job], int(reschedule_start)))
        if job in last_executed_end:
            model.Add(starts[(j, 0)] >= int(math.ceil(last_executed_end[job])))

        # Technologische Reihenfolge
        for o in range(1, len(all_ops[j])):
            model.Add(starts[(j, o)] >= ends[(j, o - 1)])

        # Deviation
        for o, (op_id, _, _) in enumerate(all_ops[j]):
            key = (job, op_id)
            if key in original_start:
                diff = model.NewIntVar(-horizon, horizon, f"diff_{j}_{o}")
                dev = model.NewIntVar(0, horizon, f"dev_{j}_{o}")
                model.Add(diff == starts[(j, o)] - original_start[key])
                model.AddAbsEquality(dev, diff)
                deviation_terms.append(dev)

    # 9. === Maschinenrestriktionen ===
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

    # 10. === Zielfunktion zusammensetzen ===
    weighted_part = model.NewIntVar(0, horizon * len(weighted_terms), "weighted_part")
    deviation_part = model.NewIntVar(0, horizon * len(deviation_terms), "deviation_part")
    model.Add(weighted_part == sum(weighted_terms))
    model.Add(deviation_part == sum(deviation_terms))

    first_op_starts = [starts[(j, 0)] for j in range(len(jobs))]
    startsum = model.NewIntVar(0, horizon * len(jobs), "startsum")
    model.Add(startsum == sum(first_op_starts))

    start_scaled = model.NewIntVar(0, horizon * len(jobs) * 100, "start_scaled")
    model.Add(start_scaled == (100 - alpha_scaled) * startsum)

    cost_main = model.NewIntVar(0, horizon * len(jobs) * 100, "cost_main")
    model.Add(cost_main == alpha_scaled * (r_scaled * weighted_part + (100 - r_scaled) * deviation_part))

    total_cost = model.NewIntVar(0, horizon * len(jobs) * 10000, "total_cost")
    model.Add(total_cost == cost_main - start_scaled)
    model.Minimize(total_cost)

    # === Solver starten ===
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.max_time_in_seconds = timeLimit
    solver.parameters.relative_gap_limit = gapRel
    status = solver.Solve(model)

    # === Lösung extrahieren ===
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        records = get_records_from_cp(
            jobs=jobs, all_ops=all_ops, starts=starts,
            solver=solver, job_column=job_column, df_times=df_times
        )
        df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", job_column, "Operation"]).reset_index(drop=True)
    else:
        print("\nSolver-Status         :", solver.StatusName(status))
        print("No feasible solution found!")
        df_schedule = pd.DataFrame()

    # === Logging ===
    print(f"\nSolver-Status         : {solver.StatusName(status)}")
    print(f"Objective Value       : {solver.ObjectiveValue():.2f}")
    print(f"Best Objective Bound  : {solver.BestObjectiveBound():.2f}")
    print(f"Laufzeit              : {solver.WallTime():.2f} Sekunden")
    print(f"Deviation terms       : {len(deviation_terms)}")

    return df_schedule


def solve_jssp_by_tardiness_and_earliness_with_devpen(
        df_jssp: pd.DataFrame, df_times: pd.DataFrame, df_executed: pd.DataFrame,
        df_original_plan: pd.DataFrame, job_column: str = "Job", earliest_start_column: str = "Arrival",
        w_t: int = 5, w_e: int = 1, r: float = 0.5,
        reschedule_start: float = 1440.0, sort_ascending: bool = False, msg: bool = False,
        timeLimit: int = 3600, gapRel: float = 0.0) -> pd.DataFrame:

    # 1. === Modell erstellen ===
    model = cp_model.CpModel()
    w_t = int(w_t)
    w_e = int(w_e)
    r_scaled = int(round(r * 100))

    # 2. === Vorverarbeitung: Ankunft, Deadline, Jobs ===
    df_times = df_times.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    earliest_start = df_times.set_index(job_column)[earliest_start_column].to_dict()
    deadline = df_times.set_index(job_column)["Deadline"].to_dict()
    jobs = df_times[job_column].tolist()

    # 3. === Schnittmenge von Operationen für Deviation bestimmen ===
    deviation_ops = set(df_jssp[[job_column, "Operation"]].apply(tuple, axis=1)) & \
                    set(df_original_plan[[job_column, "Operation"]].apply(tuple, axis=1))
    original_start = {
        (row[job_column], row["Operation"]): int(round(row["Start"]))
        for _, row in df_original_plan.iterrows()
        if (row[job_column], row["Operation"]) in deviation_ops
    }

    # 4. === Operationen gruppieren, Maschinen erfassen ===
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

    # 5. === Planungshorizont schätzen ===
    horizon = int(df_jssp["Processing Time"].sum() + max(deadline.values()))

    # 6. === Fixierte Operationen vorbereiten ===
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start]
    fixed_ops = {
        m: list(grp[["Start", "End"]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby(job_column)["End"].max().to_dict()

    # 7. === Variablen definieren ===
    starts, ends, intervals = {}, {}, {}
    weighted_terms = []
    deviation_terms = []

    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            suffix = f"{j}_{o}"
            start = model.NewIntVar(0, horizon, f"start_{suffix}")
            end = model.NewIntVar(0, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, d, end, f"interval_{suffix}")
            starts[(j, o)] = start
            ends[(j, o)] = end
            intervals[(j, o)] = (interval, m)

    # 8. === Nebenbedingungen und Zielterme ===
    for j, job in enumerate(jobs):
        last_op = len(all_ops[j]) - 1
        job_end = ends[(j, last_op)]

        # Lateness, Tardiness, Earliness
        lateness = model.NewIntVar(-horizon, horizon, f"lateness_{j}")
        model.Add(lateness == job_end - deadline[job])

        tardiness = model.NewIntVar(0, horizon, f"tardiness_{j}")
        model.AddMaxEquality(tardiness, [lateness, 0])
        term_t = model.NewIntVar(0, horizon * w_t, f"term_t_{j}")
        model.Add(term_t == w_t * tardiness)
        weighted_terms.append(term_t)

        earliness = model.NewIntVar(0, horizon, f"earliness_{j}")
        model.AddMaxEquality(earliness, [-lateness, 0])
        term_e = model.NewIntVar(0, horizon * w_e, f"term_e_{j}")
        model.Add(term_e == w_e * earliness)
        weighted_terms.append(term_e)

        # Startzeitbedingungen
        model.Add(starts[(j, 0)] >= max(earliest_start[job], int(reschedule_start)))
        if job in last_executed_end:
            model.Add(starts[(j, 0)] >= int(math.ceil(last_executed_end[job])))

        # Technologische Reihenfolge
        for o in range(1, len(all_ops[j])):
            model.Add(starts[(j, o)] >= ends[(j, o - 1)])

        # Deviation
        for o, (op_id, _, _) in enumerate(all_ops[j]):
            key = (job, op_id)
            if key in original_start:
                diff = model.NewIntVar(-horizon, horizon, f"diff_{j}_{o}")
                dev = model.NewIntVar(0, horizon, f"dev_{j}_{o}")
                model.Add(diff == starts[(j, o)] - original_start[key])
                model.AddAbsEquality(dev, diff)
                deviation_terms.append(dev)

    # 9. === Maschinenrestriktionen ===
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

    # 10. === Zielfunktion zusammensetzen ===
    weighted_part = model.NewIntVar(0, horizon * len(weighted_terms), "weighted_part")
    deviation_part = model.NewIntVar(0, horizon * len(deviation_terms), "deviation_part")
    model.Add(weighted_part == sum(weighted_terms))
    model.Add(deviation_part == sum(deviation_terms))
    total_cost = model.NewIntVar(0, horizon * len(jobs) * 100, "total_cost")
    model.Add(total_cost == r_scaled * weighted_part + (100 - r_scaled) * deviation_part)
    model.Minimize(total_cost)

    # === Solver starten ===
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.max_time_in_seconds = timeLimit
    solver.parameters.relative_gap_limit = gapRel
    status = solver.Solve(model)

    # === Lösung extrahieren ===
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        records = get_records_from_cp(
            jobs=jobs, all_ops=all_ops, starts=starts,
            solver=solver, job_column=job_column, df_times=df_times
        )
        df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", job_column, "Operation"]).reset_index(drop=True)
    else:
        print("\nSolver-Status         :", solver.StatusName(status))
        print("No feasible solution found!")
        df_schedule = pd.DataFrame()

    # === Logging ===
    print(f"\nSolver-Status         : {solver.StatusName(status)}")
    print(f"Objective Value       : {solver.ObjectiveValue():.2f}")
    print(f"Best Objective Bound  : {solver.BestObjectiveBound():.2f}")
    print(f"Laufzeit              : {solver.WallTime():.2f} Sekunden")
    print(f"Deviation terms       : {len(deviation_terms)}")

    return df_schedule



def solve_jssp_by_tardiness_and_earliness_with_fixed_ops(
        df_jssp: pd.DataFrame, df_times: pd.DataFrame, df_executed: pd.DataFrame,
        w_t: int = 5, w_e: int = 1, reschedule_start: float = 1440.0,
        job_column: str = "Job", earliest_start_column: str = "Arrival", sort_ascending: bool = False,
        msg: bool = False, timeLimit: int = 3600, gapRel: float = 0.0) -> pd.DataFrame:
    """
    Löst ein Job-Shop-Scheduling-Problem mit gewichteter Tardiness und Earliness
    unter Berücksichtigung fest eingeplanter Operationen. Verwendet Constraint Programming.
    """

    model = cp_model.CpModel()

    # 1. Vorbereitung: Gewichte in ganze Zahlen wandeln
    w_t = int(w_t)
    w_e = int(w_e)

    # 2. Arrival/Deadline extrahieren
    df_times = df_times.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    earliest_start = df_times.set_index(job_column)[earliest_start_column].to_dict()
    deadline = df_times.set_index(job_column)["Deadline"].to_dict()
    jobs = df_times[job_column].tolist()

    # 3. Operationen pro Job strukturieren, Maschinenmenge bestimmen
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

    # 4. Grober Planungshorizont
    horizon = int(df_jssp["Processing Time"].sum() + max(deadline.values()))

    # 5. Fixierte Operationen aus df_executed extrahieren
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start]
    fixed_ops = {
        m: list(grp[["Start", "End"]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }

    # 6. Variablen anlegen
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

    # 7. Zielbedingung je Job (lateness → earliness/tardiness → Gewichtung)
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

        # Ankunfts- und Rescheduling-Bedingungen
        model.Add(starts[(j, 0)] >= earliest_start[job])
        model.Add(starts[(j, 0)] >= int(reschedule_start))

        if job in df_executed[job_column].values:
            last_fixed_end = df_executed[df_executed[job_column] == job]["End"].max()
            model.Add(starts[(j, 0)] >= int(math.ceil(last_fixed_end)))

        # Technologische Reihenfolge
        for o in range(1, len(all_ops[j])):
            model.Add(starts[(j, o)] >= ends[(j, o - 1)])

    # 8. Maschinenrestriktionen (inkl. fixer Intervalle)
    for m in machines:
        machine_intervals = [interval for (j, o), (interval, mach) in intervals.items() if mach == m]
        for fixed_start, fixed_end in fixed_ops.get(m, []):
            start = math.floor(fixed_start)
            end = math.ceil(fixed_end)
            duration = end - start
            fixed_interval = model.NewIntervalVar(start, duration, end, f"fixed_{m}_{end}")
            machine_intervals.append(fixed_interval)
        model.AddNoOverlap(machine_intervals)

    # 9. Zielfunktion minimieren
    model.Minimize(sum(weighted_terms))

    # 10. Solver starten
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.relative_gap_limit = gapRel
    solver.parameters.max_time_in_seconds = timeLimit
    status = solver.Solve(model)

    # Ergebnis extrahieren
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        records = get_records_from_cp(
            jobs=jobs, all_ops=all_ops, starts=starts,
            solver=solver, job_column=job_column, df_times=df_times
        )
        df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", job_column, "Operation"]).reset_index(drop=True)
    else:
        print(f"\nSolver-Status         : {solver.StatusName(status)}")
        print("No feasible solution found!")
        df_schedule = pd.DataFrame()

    # Logging
    print(f"\nSolver-Status         : {solver.StatusName(status)}")
    print(f"Objective Value       : {solver.ObjectiveValue():.2f}")
    print(f"Best Objective Bound  : {solver.BestObjectiveBound():.2f}")
    print(f"Laufzeit              : {solver.WallTime():.2f} Sekunden")

    return df_schedule


