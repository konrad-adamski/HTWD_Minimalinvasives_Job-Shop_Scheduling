from src.models.cp.builder import get_records_from_cp
from fractions import Fraction

import math
import pandas as pd
from ortools.sat.python import cp_model

def solve_jssp_lateness_with_start_deviation_OLD(
        df_jssp: pd.DataFrame, df_times: pd.DataFrame, df_original_plan: pd.DataFrame | None,
        df_active: pd.DataFrame | None = None, job_column: str = "Job", earliest_start_column: str = "Arrival",
        w_t: int = 5, w_e: int = 1, w_first: int = 1, main_pct: float = 0.5,
        schedule_start: float = 1440.0, sort_ascending: bool | None = False, msg: bool = False,
        timeLimit: int = 3600, gapRel: float = 0.0) -> pd.DataFrame:

    # 1. === Modell erstellen ===
    model = cp_model.CpModel()
    w_t = int(w_t)
    w_e = int(w_e)
    w_first = int(w_first)

    if df_original_plan is None:
        print("Es liegt kein ursprünglicher Schedule vor!")
        main_pct = 1
    main_pct_frac = Fraction(main_pct).limit_denominator(100)

    numerator = main_pct_frac.numerator
    denominator = main_pct_frac.denominator

    main_factor = numerator
    dev_factor = denominator - numerator

    # 2. === Vorverarbeitung: Ankunft, Deadline, Jobs ===
    if sort_ascending is not None:
        df_times = df_times.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)

    earliest_start = df_times.set_index(job_column)[earliest_start_column].to_dict()
    deadline = df_times.set_index(job_column)["Deadline"].to_dict()
    jobs = df_times[job_column].tolist()

    # 3. === Schnittmenge von Operationen für Deviation bestimmen (optional) ===
    # Wenn ein ursprünglicher Plan vorhanden ist, vergleiche Operationen mit dem aktuellen Plan
    # und speichere Startzeiten, um Deviation-Terme zu ermöglichen.
    if df_original_plan is not None:
        deviation_ops = set(df_jssp[[job_column, "Operation"]].apply(tuple, axis=1)) & \
                        set(df_original_plan[[job_column, "Operation"]].apply(tuple, axis=1))
        original_start = {
            (row[job_column], row["Operation"]): int(round(row["Start"]))
            for _, row in df_original_plan.iterrows()
            if (row[job_column], row["Operation"]) in deviation_ops
        }

    else:
        # Kein ursprünglicher Plan vorhanden: keine Deviation möglich
        original_start = {}

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

    # 6. === Fixierte Operationen vorbereiten (nur wenn vorhanden) ===
    if df_active is not None and not df_active.empty:
        df_active_fixed = df_active[df_active["End"] >= schedule_start]
        fixed_ops = {
            m: list(grp[["Start", "End"]].itertuples(index=False, name=None))
            for m, grp in df_active_fixed.groupby("Machine")
        }
        last_executed_end = df_active.groupby(job_column)["End"].max().to_dict()
    else:
        fixed_ops = {}
        last_executed_end = {}

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
        model.Add(starts[(j, 0)] >= max(earliest_start[job], int(schedule_start)))
        if job in last_executed_end:
            model.Add(starts[(j, 0)] >= int(math.ceil(last_executed_end[job])))

        # Technologische Reihenfolge
        for o in range(1, len(all_ops[j])):
            model.Add(starts[(j, o)] >= ends[(j, o - 1)])

        # Deviation zur ursprünglichen Planung (wenn vorhanden)
        if original_start:
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
    model.Add(weighted_part == sum(weighted_terms))

    deviation_part = model.NewIntVar(0, horizon * len(deviation_terms), "deviation_part")
    model.Add(deviation_part == sum(deviation_terms))

    first_op_starts = [starts[(j, 0)] for j in range(len(jobs))]
    startsum = model.NewIntVar(0, horizon * len(jobs), "startsum")

    # Startsumme verschoben um Schedule-Start
    model.Add(startsum == sum(first_op_starts) - int(schedule_start * len(jobs)))

    first_op_delay = model.NewIntVar(0, horizon * len(jobs) * w_first, "first_op_delay")
    model.Add(first_op_delay == w_first * startsum)

    combined_lateness = model.NewIntVar(-horizon * len(jobs) * 100, horizon * len(jobs) * 100, "combined_lateness")
    model.Add(combined_lateness == weighted_part - first_op_delay)

    scaled_lateness = model.NewIntVar(-10000000, 10000000, "scaled_lateness")
    model.Add(scaled_lateness == main_factor * combined_lateness)

    deviation_penalty = model.NewIntVar(0, 10000000, "deviation_penalty")
    model.Add(deviation_penalty == dev_factor * deviation_part)

    total_cost = model.NewIntVar(-100000000, 100000000, "total_cost")
    model.Add(total_cost == scaled_lateness + deviation_penalty)
    model.Minimize(total_cost)

    # 11. === Solver starten ===
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.max_time_in_seconds = timeLimit
    solver.parameters.relative_gap_limit = gapRel
    status = solver.Solve(model)

    # 12. === Lösung extrahieren ===
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

    # 13. === Logging ===
    print(f"\nSolver-Status         : {solver.StatusName(status)}")
    print(f"Objective Value       : {solver.ObjectiveValue():.2f}")
    print(f"Best Objective Bound  : {solver.BestObjectiveBound():.2f}")
    print(f"Laufzeit              : {solver.WallTime():.2f} Sekunden")
    print(f"Deviation terms       : {len(deviation_terms)}")

    return df_schedule

from src.models.cp.builder import get_records_from_cp
from fractions import Fraction

import math
import pandas as pd
from ortools.sat.python import cp_model

from src.models.cp.builder import get_records_from_cp
from fractions import Fraction

import math
import pandas as pd
from ortools.sat.python import cp_model

from src.models.cp.builder import get_records_from_cp
from fractions import Fraction

import math
import pandas as pd
from ortools.sat.python import cp_model

def solve_jssp_lateness_with_start_deviation(
        df_jssp: pd.DataFrame, df_times: pd.DataFrame, df_original_plan: pd.DataFrame | None,
        df_active: pd.DataFrame | None = None, job_column: str = "Job", earliest_start_column: str = "Arrival",
        w_t: int = 5, w_e: int = 1, w_first: int = 1, main_pct: float = 0.5, latest_start_buffer: int = 360,
        schedule_start: float = 1440.0, sort_ascending: bool | None = False, msg: bool = False,
        timeLimit: int = 3600, gapRel: float = 0.0) -> pd.DataFrame:

    # 1. === Modell erstellen ===
    model = cp_model.CpModel()
    w_t = int(w_t)
    w_e = int(w_e)
    w_first = int(w_first)

    if df_original_plan is None:
        print("Es liegt kein ursprünglicher Schedule vor!")
        main_pct = 1
    main_pct_frac = Fraction(main_pct).limit_denominator(100)

    numerator = main_pct_frac.numerator
    denominator = main_pct_frac.denominator

    main_factor = numerator
    dev_factor = denominator - numerator

    # 2. === Vorverarbeitung: Ankunft, Deadline, Jobs ===
    if sort_ascending is not None:
        df_times = df_times.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)

    earliest_start = df_times.set_index(job_column)[earliest_start_column].to_dict()
    deadline = df_times.set_index(job_column)["Deadline"].to_dict()
    jobs = df_times[job_column].tolist()

    # 3. === Schnittmenge von Operationen für Deviation bestimmen (optional) ===
    if df_original_plan is not None:
        deviation_ops = set(df_jssp[[job_column, "Operation"]].apply(tuple, axis=1)) & \
                        set(df_original_plan[[job_column, "Operation"]].apply(tuple, axis=1))
        original_start = {
            (row[job_column], row["Operation"]): int(round(row["Start"]))
            for _, row in df_original_plan.iterrows()
            if (row[job_column], row["Operation"]) in deviation_ops
        }
    else:
        original_start = {}

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
    if df_active is not None and not df_active.empty:
        df_active_fixed = df_active[df_active["End"] >= schedule_start]
        fixed_ops = {
            m: list(grp[["Start", "End"]].itertuples(index=False, name=None))
            for m, grp in df_active_fixed.groupby("Machine")
        }
        last_executed_end = df_active.groupby(job_column)["End"].max().to_dict()
    else:
        fixed_ops = {}
        last_executed_end = {}

    # 7. === Variablen definieren ===
    starts, ends, intervals = {}, {}, {}
    weighted_terms = []
    deviation_terms = []
    first_delay_penalties = []

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
        model.Add(starts[(j, 0)] >= max(earliest_start[job], int(schedule_start)))
        if job in last_executed_end:
            model.Add(starts[(j, 0)] >= int(math.ceil(last_executed_end[job])))

        # Technologische Reihenfolge
        for o in range(1, len(all_ops[j])):
            model.Add(starts[(j, o)] >= ends[(j, o - 1)])

        # Deviation zur ursprünglichen Planung
        if original_start:
            for o, (op_id, _, _) in enumerate(all_ops[j]):
                key = (job, op_id)
                if key in original_start:
                    diff = model.NewIntVar(-horizon, horizon, f"diff_{j}_{o}")
                    dev = model.NewIntVar(0, horizon, f"dev_{j}_{o}")
                    model.Add(diff == starts[(j, o)] - original_start[key])
                    model.AddAbsEquality(dev, diff)
                    deviation_terms.append(dev)

        # Frühstart-Strafe = Belohnung für spätes Starten
        first_start = starts[(j, 0)]
        total_processing_time = sum(d for (_, _, d) in all_ops[j])
        latest_desired_start = deadline[job] - total_processing_time - latest_start_buffer

        early_penalty = model.NewIntVar(0, horizon, f"early_penalty_{j}")
        model.AddMaxEquality(early_penalty, [latest_desired_start - first_start, 0])

        term_first = model.NewIntVar(0, horizon * w_first, f"term_first_{j}")
        model.Add(term_first == w_first * early_penalty)
        first_delay_penalties.append(term_first)

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
    model.Add(weighted_part == sum(weighted_terms))

    deviation_part = model.NewIntVar(0, horizon * len(deviation_terms), "deviation_part")
    model.Add(deviation_part == sum(deviation_terms))

    first_op_delay = model.NewIntVar(0, horizon * len(jobs) * w_first, "first_op_delay")
    model.Add(first_op_delay == sum(first_delay_penalties))

    combined_lateness = model.NewIntVar(-horizon * len(jobs) * 100, horizon * len(jobs) * 100, "combined_lateness")
    model.Add(combined_lateness == weighted_part + first_op_delay)

    scaled_lateness = model.NewIntVar(-10000000, 10000000, "scaled_lateness")
    model.Add(scaled_lateness == main_factor * combined_lateness)

    deviation_penalty = model.NewIntVar(0, 10000000, "deviation_penalty")
    model.Add(deviation_penalty == dev_factor * deviation_part)

    total_cost = model.NewIntVar(-100000000, 100000000, "total_cost")
    model.Add(total_cost == scaled_lateness + deviation_penalty)
    model.Minimize(total_cost)

    # 11. === Solver starten ===
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.max_time_in_seconds = timeLimit
    solver.parameters.relative_gap_limit = gapRel
    status = solver.Solve(model)

    # 12. === Lösung extrahieren ===
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

    # 13. === Logging ===
    print(f"\nSolver-Status         : {solver.StatusName(status)}")
    print(f"Objective Value       : {solver.ObjectiveValue():.2f}")
    print(f"Best Objective Bound  : {solver.BestObjectiveBound():.2f}")
    print(f"Laufzeit              : {solver.WallTime():.2f} Sekunden")
    print(f"Deviation terms       : {len(deviation_terms)}")

    return df_schedule


