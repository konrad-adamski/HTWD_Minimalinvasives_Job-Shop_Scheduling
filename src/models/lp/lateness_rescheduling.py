import math
import pulp
import pandas as pd
import time


# Lateness Rescheduling----------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------

# Min. Summe abs. Lateness -----------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------

# mit Deviation Penalty (& fixierte Operation, die hineinlaufen)
def solve_jssp_sum_with_devpen(df_jssp: pd.DataFrame, df_times: pd.DataFrame, df_executed: pd.DataFrame,
                                 df_original_plan: pd.DataFrame, r: float = 0.5, reschedule_start: float = 1440.0,
                                 job_column: str = "Job", earliest_start_column: str = "Arrival",
                                 solver: str = "HiGHS", epsilon: float = 0.0, var_cat: str = "Continuous", 
                                 time_limit: int | None = 10800, sort_ascending: bool = False,
                                 **solver_args) -> pd.DataFrame:
    """
    Minimiert: Z(σ) = r * sum_j [|C_j - d_j|] + (1 - r) * Summe aller Startzeitabweichungen zum Originalplan
    Berücksichtigt technologische Reihenfolge, Maschinenkonflikte, Fixierungen und Planabweichung.
    """
    start_time = time.time()

    # 1. Vorverarbeitung & Struktur
    jobs, all_ops, machines, earliest_start, deadline = prepare_jssp_inputs(
        df_jssp, df_times, job_column, earliest_start_column, sort_ascending
    )

    original_start = {
        (row[job_column], row["Operation"]): row["Start"]
        for _, row in df_original_plan.iterrows()
    }

    # 2. Fixierte Operationen
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start]
    fixed_ops = {
        m: list(g[["Start", "End", job_column]].itertuples(index=False, name=None))
        for m, g in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby(job_column)["End"].max().to_dict()

    # 3. Big-M
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(earliest_start.values())
    max_deadline = max(deadline.values())
    num_machines = len(machines)
    bigM = math.ceil((max_deadline - min_arrival + sum_proc_time / math.sqrt(num_machines)) / 1000) * 1000
    print(f"BigM: {bigM}")

    # 4. Variablen
    starts, ends, abs_lateness = build_jssp_variables_with_fixed_ops(
        jobs, all_ops, earliest_start, last_executed_end, reschedule_start, var_cat,
        abs_lateness={"lowBound": 0, "cat": var_cat}
    )

    deviation_vars = {}
    prob = pulp.LpProblem("JSSP_AbsLateness_Deviation", pulp.LpMinimize)

    # 5. Ziel: Kombination aus Lateness und Abweichung zum Originalplan
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        prob += ends[j] == starts[(j, len(seq)-1)] + seq[-1][2]
        lateness = ends[j] - deadline[job]
        prob += abs_lateness[j] >= lateness
        prob += abs_lateness[j] >= -lateness

        for o, (op_id, _, _) in enumerate(seq):
            key = (job, op_id)
            if key in original_start:
                dev = pulp.LpVariable(f"dev_{j}_{o}", lowBound=0, cat=var_cat)
                deviation_vars[(j, o)] = dev
                prob += dev >= starts[(j, o)] - original_start[key]
                prob += dev >= original_start[key] - starts[(j, o)]

    prob += r * 10 * pulp.lpSum(abs_lateness.values()) + (1 - r) * 10 * pulp.lpSum(deviation_vars.values())

    # 6. Technologische Reihenfolge
    define_technological_constraints_with_fixed_ops(
        prob, jobs, all_ops, starts, ends, abs_lateness, deadline,
        last_executed_end, earliest_start, reschedule_start, mode="absolute_lateness"
    )

    # 7. Maschinenkonflikte inkl. Fixblöcke
    add_machine_constraints_with_fixed_ops(prob, all_ops, starts, machines, epsilon, bigM, fixed_ops)

    # 8. Solver starten
    solver_instance = get_solver_instance(solver, time_limit, solver_args)
    prob.solve(solver_instance)
    objective_value = pulp.value(prob.objective)

    # 9. Ergebnisse extrahieren
    df_schedule = get_records_df(df_jssp, df_times, jobs, starts, job_column)
    df_schedule["Lateness"] = (df_schedule["End"] - df_schedule["Deadline"]).round(2)
    df_schedule["Absolute Lateness"] = df_schedule["Lateness"].abs().round(2)
    df_schedule = df_schedule.sort_values(["Start", job_column, "Operation"]).reset_index(drop=True)

    # 10. Logging
    print("\nSolver-Informationen:")
    print(f"  Zielfunktionswert         : {round(objective_value, 4)}")
    print(f"  Solver-Status             : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen          : {len(prob.variables())}")
    print(f"  Anzahl Constraints        : {len(prob.constraints)}")
    print(f"  Laufzeit                  : ~{time.time() - start_time:.0f} Sekunden")

    return df_schedule



# einfach (nur fixierte Opertion, die hineinlaufen)
def solve_jssp_sum_with_fixed_ops(df_jssp: pd.DataFrame, df_times: pd.DataFrame, df_executed: pd.DataFrame,
                                   reschedule_start: float = 1440.0, job_column: str = "Job", 
                                   earliest_start_column: str = "Arrival",
                                   solver: str = "HiGHS", epsilon: float = 0.0, var_cat: str = "Continuous", 
                                   time_limit: int | None = 10800, sort_ascending: bool = False,
                                   **solver_args) -> pd.DataFrame:
    """
    Minimiert die Summe der absoluten Lateness unter Berücksichtigung bereits ausgeführter Operationen.
    """
    import time, math, pulp
    start_time = time.time()

    # 1. Vorverarbeitung & Struktur
    jobs, all_ops, machines, earliest_start, deadline = prepare_jssp_inputs(
        df_jssp, df_times, job_column, earliest_start_column, sort_ascending
    )

    # 2. Fixierte Operationen
    df_executed_fixed = df_executed[df_executed["End"] >= reschedule_start]
    fixed_ops = {
        m: list(grp[["Start", "End", job_column]].itertuples(index=False, name=None))
        for m, grp in df_executed_fixed.groupby("Machine")
    }
    last_executed_end = df_executed.groupby(job_column)["End"].max().to_dict()

    # 3. Big-M Berechnung
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(earliest_start.values())
    max_deadline = max(deadline.values())
    num_machines = len(machines)
    bigM = math.ceil((max_deadline - min_arrival + sum_proc_time / math.sqrt(num_machines)) / 1000) * 1000
    print(f"BigM: {bigM}")

    # 4. Variablen
    starts, ends, abs_lateness = build_jssp_variables_with_fixed_ops(
        jobs, all_ops, earliest_start, last_executed_end, reschedule_start, var_cat,
        abs_lateness={"lowBound": 0, "cat": var_cat}
    )

    # 5. Modell
    prob = pulp.LpProblem("JSSP_SumAbsLateness_FixedOps", pulp.LpMinimize)
    prob += pulp.lpSum(abs_lateness.values())

    # 6. Constraints
    define_technological_constraints_with_fixed_ops(
        prob, jobs, all_ops, starts, ends, abs_lateness, deadline,
        last_executed_end, earliest_start, reschedule_start,
        mode="absolute_lateness"
    )

    add_machine_constraints_with_fixed_ops(prob, all_ops, starts, machines, epsilon, bigM, fixed_ops)

    # 7. Solver & Lösung
    solver_instance = get_solver_instance(solver, time_limit, solver_args)
    prob.solve(solver_instance)
    objective_value = pulp.value(prob.objective)

    # 8. Ergebnisaufbereitung
    df_schedule = get_records_df(df_jssp, df_times, jobs, starts, job_column)
    df_schedule["Lateness"] = (df_schedule["End"] - df_schedule["Deadline"]).round(2)
    df_schedule["Absolute Lateness"] = df_schedule["Lateness"].abs().round(2)
    df_schedule = df_schedule.sort_values(["Start", job_column, "Operation"]).reset_index(drop=True)

    # 9. Logging
    print("\nSolver-Informationen:")
    print(f"  Summe absolute Lateness  : {round(objective_value, 4)}")
    print(f"  Solver-Status            : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen         : {len(prob.variables())}")
    print(f"  Anzahl Constraints       : {len(prob.constraints)}")
    print(f"  Laufzeit                 : ~{time.time() - start_time:.0f} Sekunden")

    return df_schedule


