from src.models.lp.solver_builder import *
import time
import math
import pulp
import pandas as pd


# Tardiness Rescheduling with Arrivals & Deadline ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------

# Min. Summe Tardiness -------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# mit Deviation Penalty (& fixierte Operation, die hineinlaufen)
def solve_jssp_sum_with_devpen(df_jssp: pd.DataFrame, df_times: pd.DataFrame, df_executed: pd.DataFrame,
                               df_original_plan: pd.DataFrame, r: float = 0.5, reschedule_start: float = 1440.0,
                               job_column: str = "Job", earliest_start_column: str = "Arrival",
                               solver: str = "HiGHS", epsilon: float = 0.0, var_cat: str = "Continuous",
                               time_limit: int | None = 10800, sort_ascending: bool = False,
                               **solver_args) -> pd.DataFrame:
    """
    Minimiert eine gewichtete Summe aus Tardiness und Abweichung vom ursprünglichen Plan
    unter Berücksichtigung bereits ausgeführter Operationen.

    Zielfunktion: Z(σ) = r * Sum_Tardiness + (1 - r) * Sum_Deviation
    """

    start_time = time.time()

    # 1. Vorverarbeitung & Inputs
    jobs, all_ops, machines, arrival, deadline = prepare_jssp_inputs(
        df_jssp, df_times, job_column, earliest_start_column, sort_ascending
    )

    n = len(jobs)

    original_start = {
        (row[job_column], row["Operation"]): row["Start"]
        for _, row in df_original_plan.iterrows()
    }

    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    fixed_ops = {
        m: list(grp[["Start", "End", job_column]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby(job_column)["End"].max().to_dict()

    # 2. Variablen erzeugen
    starts, ends, tards = build_jssp_variables_with_fixed_ops(
        jobs, all_ops, arrival, last_executed_end, reschedule_start, var_cat,
        tards={"lowBound": 0, "cat": var_cat}
    )

    deviation_vars = {}
    for j in range(n):
        for o, (op_id, _, _) in enumerate(all_ops[j]):
            key = (jobs[j], op_id)
            if key in original_start:
                dev = pulp.LpVariable(f"dev_{j}_{o}", lowBound=0, cat=var_cat)
                deviation_vars[(j, o)] = dev

    # 3. Modellaufbau
    prob = pulp.LpProblem("JSSP_Weighted_Deviation", pulp.LpMinimize)
    prob += r * 10 * pulp.lpSum(tards.values()) + (1 - r) * 10 * pulp.lpSum(deviation_vars.values())

    # 4. Technologische Constraints + Tardiness
    define_technological_constraints_with_fixed_ops(
        prob, jobs, all_ops, starts, ends, tards, deadline,
        last_executed_end, arrival, reschedule_start, mode="tardiness"
    )

    # 5. Abweichungs-Constraints
    for (j, o), dev in deviation_vars.items():
        key = (jobs[j], all_ops[j][o][0])
        original = original_start[key]
        prob += dev >= starts[(j, o)] - original
        prob += dev >= original - starts[(j, o)]

    # 6. Maschinenkonflikte + Fixe
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_deadline = max(deadline.values())
    bigM = math.ceil((max_deadline - min_arrival + sum_proc_time / max(1, math.sqrt(len(machines)))) / 1000) * 1000
    print(f"BigM: {bigM}")

    add_machine_constraints_with_fixed_ops(prob, all_ops, starts, machines, epsilon, bigM, fixed_ops)

    # 7. Lösen
    solver_instance = get_solver_instance(solver, time_limit, solver_args)
    prob.solve(solver_instance)
    objective_value = pulp.value(prob.objective)

    # 8. Ergebnisse
    df_schedule = get_records_df(df_jssp, df_times, jobs, starts, job_column=job_column)
    df_schedule["Tardiness"] = (df_schedule["End"] - df_schedule["Deadline"]).clip(lower=0).round(2)
    df_schedule["Deviation"] = df_schedule.apply(
        lambda row: round(abs(row["Start"] - original_start.get((row[job_column], row["Operation"]), row["Start"])), 2),
        axis=1
    )

    # 9. Logging
    print("\nSolver-Informationen:")
    print(f"  Zielfunktionswert       : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")
    print(f"  Laufzeit                : ~{time.time() - start_time:.0f} Sekunden")

    return df_schedule



# einfach (nur fixierte Opertion, die hineinlaufen)
def solve_jssp_sum_with_fixed_ops(df_jssp: pd.DataFrame, df_times: pd.DataFrame, df_executed: pd.DataFrame,
                                   reschedule_start: float = 1440.0, job_column: str = "Job",
                                   earliest_start_column: str = "Arrival",
                                   solver: str = "HiGHS", epsilon: float = 0.0, var_cat: str = "Continuous",
                                   time_limit: int | None = 10800, sort_ascending: bool = False,
                                   **solver_args) -> pd.DataFrame:
    """
    Minimiert die Summe der Tardiness (Verspätung) aller Jobs unter Berücksichtigung bereits ausgeführter Operationen.

    Fixierte Operationen (df_executed) blockieren ihre Zeitfenster. Neue Operationen werden ab reschedule_start neu eingeplant.
    """

    import time
    start_time = time.time()

    # 1. Vorverarbeitung
    jobs, all_ops, machines, earliest_start, deadline = prepare_jssp_inputs(
        df_jssp, df_times, job_column, earliest_start_column, sort_ascending
    )

    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    fixed_ops = {
        m: list(grp[["Start", "End", job_column]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby(job_column)["End"].max().to_dict()

    # 2. Big-M zur Konfliktmodellierung
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_start = min(earliest_start.values())
    max_deadline = max(deadline.values())
    bigM = math.ceil((max_deadline - min_start + sum_proc_time / max(len(machines), 1)) / 1000) * 1000
    print(f"BigM: {bigM}")

    # 3. Modell und Variablen
    prob = pulp.LpProblem("JSSP_SumTardiness_Fixed", pulp.LpMinimize)

    starts, ends, tards = build_jssp_variables_with_fixed_ops(
        jobs, all_ops, earliest_start, last_executed_end, reschedule_start, var_cat,
        tards={"lowBound": 0, "cat": var_cat}
    )

    prob += pulp.lpSum(tards.values())

    # 4. Technologische Reihenfolge + Tardiness-Ziel
    define_technological_constraints_with_fixed_ops(
        prob, jobs, all_ops, starts, ends, tards, deadline,
        last_executed_end, earliest_start, reschedule_start, mode="tardiness"
    )

    # 5. Maschinenkonflikte mit fixierten Operationen
    add_machine_constraints_with_fixed_ops(
        prob, all_ops, starts, machines, epsilon, bigM, fixed_ops
    )

    # 6. Solver konfigurieren & starten
    solver_instance = get_solver_instance(solver, time_limit, solver_args)
    prob.solve(solver_instance)
    objective_value = pulp.value(prob.objective)

    # 7. Ergebnis aufbereiten
    df_schedule = get_records_df(
        df_jssp=df_jssp,
        df_times=df_times,
        jobs_list=jobs,
        starts=starts,
        job_column=job_column
    )
    df_schedule["Tardiness"] = (df_schedule["End"] - df_schedule["Deadline"]).clip(lower=0).round(2)
    df_schedule = df_schedule.sort_values(["Start", job_column, "Operation"]).reset_index(drop=True)

    # 8. Logging
    print("\nSolver-Informationen:")
    print(f"  Summe Tardiness         : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")
    print(f"  Laufzeit                : ~{time.time() - start_time:.0f} Sekunden")

    return df_schedule


# Min. Max Tardiness ---------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# mit Deviation Penalty (& fixierte Operation, die hineinlaufen)
def solve_jssp_max_with_devpen(df_jssp: pd.DataFrame, df_times: pd.DataFrame, df_executed: pd.DataFrame,
                                         df_original_plan: pd.DataFrame, r: float = 0.5, reschedule_start: float = 1440.0,
                                         job_column: str = "Job", earliest_start_column: str = "Arrival",
                                         solver: str = "HiGHS", epsilon: float = 0.0, var_cat: str = "Continuous",
                                         time_limit: int | None = 10800, sort_ascending: bool = False,
                                         **solver_args) -> pd.DataFrame:
    """
    Minimiert eine bikriterielle Zielfunktion:
    Maximale Tardiness + Abweichung vom ursprünglichen Plan (weighted sum).

    Zielfunktion: Z(σ) = r * max_j Tardiness_j + (1 - r) * D(σ)
    """
    import time, pulp
    start_time = time.time()

    # 1. Vorverarbeitung & Inputs
    jobs, all_ops, machines, arrival, deadline = prepare_jssp_inputs(
        df_jssp, df_times, job_column, earliest_start_column, sort_ascending
    )

    original_start = {
        (row[job_column], row["Operation"]): row["Start"]
        for _, row in df_original_plan.iterrows()
    }

    # 2. BigM & Fixblöcke vorbereiten
    bigM = get_bigM_estimate(df_jssp, arrival, deadline)
    fixed_ops, last_executed_end = get_fixed_ops(df_executed, reschedule_start, job_column)

    # 3. Modell
    prob = pulp.LpProblem("JSSP_MaxTardiness_DevPen", pulp.LpMinimize)
    starts, ends = build_jssp_variables_with_fixed_ops(
        jobs, all_ops, arrival, last_executed_end, reschedule_start, var_cat
    )
    tard = {j: pulp.LpVariable(f"tardiness_{j}", lowBound=0, cat=var_cat) for j in range(len(jobs))}
    max_tard = pulp.LpVariable("max_tardiness", lowBound=0, cat=var_cat)

    # 4. Planabweichungen
    deviation_vars = {}
    for j, job in enumerate(jobs):
        for o, (op_id, _, _) in enumerate(all_ops[j]):
            key = (job, op_id)
            if key in original_start:
                dev = pulp.LpVariable(f"dev_{j}_{o}", lowBound=0, cat=var_cat)
                deviation_vars[(j, o)] = dev
                prob += dev >= starts[(j, o)] - original_start[key]
                prob += dev >= original_start[key] - starts[(j, o)]

    # 5. Zielfunktion
    prob += r * 10 * max_tard + (1 - r) * 10 * pulp.lpSum(deviation_vars.values())

    # 6. Technologische Constraints & Tardiness
    define_technological_constraints_with_fixed_ops(
        prob, jobs, all_ops, starts, ends, tard, deadline, last_executed_end, reschedule_start, mode="tardiness"
    )
    for j in range(len(jobs)):
        prob += max_tard >= tard[j]

    # 7. Maschinenkonflikte
    add_machine_constraints_with_fixed_ops(
        prob, jobs, all_ops, machines, starts, fixed_ops, bigM, epsilon
    )

    # 8. Solver starten
    solver_args.setdefault("msg", True)
    solver_args.setdefault("timeLimit", time_limit)
    cmd = get_solver_instance(solver, solver_args)
    prob.solve(cmd)
    objective_value = pulp.value(prob.objective)

    # 9. Ergebnis
    df_schedule = get_records_df(
        jobs, all_ops, starts, arrival, deadline, job_column, df_times
    )

    # 10. Logging
    print("\nSolver-Informationen:")
    print(f"  Zielfunktionswert       : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")
    print(f"  Laufzeit                : ~{time.time() - start_time:.0f} Sekunden")

    return df_schedule


# einfach (nur fixierte Opertion, die hineinlaufen)
def solve_jssp_max_with_fixed_ops(df_jssp: pd.DataFrame, df_times: pd.DataFrame, df_executed: pd.DataFrame,
                                   reschedule_start: float = 1440.0, job_column: str = "Job", earliest_start_column: str = "Arrival",
                                   solver: str = "HiGHS", epsilon: float = 0.0,
                                   var_cat: str = "Continuous", time_limit: int | None = 10800,
                                   sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert die maximale Tardiness unter allen Jobs mit fixierten Operationen.
    """
    import time, math, pulp
    start_time = time.time()

    # 1. Vorbereitung
    jobs, all_ops, machines, earliest_start, deadline = prepare_jssp_inputs(df_jssp, df_times, job_column, 
                                                                            earliest_start_column, sort_ascending
                                                                           )

    # 2. Fixierte Operationen
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start].copy()
    fixed_ops = {
        m: list(grp[["Start", "End", job_column]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby(job_column)["End"].max().to_dict()

    # 3. BigM
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(earliest_start.values())
    max_deadline = max(deadline.values())
    bigM = math.ceil((max_deadline - min_arrival + sum_proc_time / math.sqrt(len(machines))) / 1000) * 1000
    print(f"BigM: {bigM}")

    # 4. Modell & Variablen
    prob = pulp.LpProblem("JSSP_MaxTardiness_Fixed", pulp.LpMinimize)

    starts, ends, tards, max_tard = build_jssp_variables_with_fixed_ops(
        jobs, all_ops, earliest_start, last_executed_end, reschedule_start, var_cat,
        tards={"lowBound": 0, "cat": var_cat},
        max_tard={"lowBound": 0, "cat": var_cat}
    )

    prob += max_tard

    # 5. Technologische Reihenfolge und Tardiness-Definition
    define_technological_constraints_with_fixed_ops(
        prob, jobs, all_ops, starts, ends, tards, deadline,
        last_executed_end, earliest_start, reschedule_start, mode="tardiness"
    )

    for j in range(len(jobs)):
        prob += max_tard >= tards[j]

    # 6. Maschinenkonflikte inkl. Fixe
    add_machine_constraints_with_fixed_ops(prob, all_ops, starts, machines, epsilon, bigM, fixed_ops)

    # 7. Lösen
    solver_instance = get_solver_instance(solver, time_limit, solver_args)
    prob.solve(solver_instance)
    objective_value = pulp.value(prob.objective)

    # 8. Ergebnis
    df_schedule = get_records_df(df_jssp, df_times, jobs, starts, job_column=job_column)
    df_schedule["Tardiness"] = (df_schedule["End"] - df_schedule["Deadline"]).clip(lower=0).round(2)
    df_schedule = df_schedule.sort_values([job_column, "Operation"]).reset_index(drop=True)

    # 9. Logging
    print("\nSolver-Informationen:")
    print(f"  Maximale Tardiness      : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")
    print(f"  Laufzeit                : ~{time.time() - start_time:.0f} Sekunden")

    return df_schedule


