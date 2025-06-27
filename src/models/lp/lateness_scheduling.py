from src.models.lp.solver_builder import *
import math
import pulp
import pandas as pd
import time

# Lateness Scheduling -----------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# - Summe Absolute Lateness
# - Max Absolute Lateness


# Min. Summe Absolute Lateness ----------------------------------------------------------------------------------------

def solve_jssp_sum(df_jssp: pd.DataFrame, df_times: pd.DataFrame, job_column: str = "Job", earliest_start_column: str = "Arrival",
                   solver: str = 'HiGHS', epsilon: float = 0.0, var_cat: str = "Continuous", 
                   time_limit: int | None = 10800, sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    start_time = time.time()

    # 1. Vorverarbeitung
    jobs, all_ops, machines, arrival, deadline = prepare_jssp_inputs(
        df_jssp, df_times, job_column, earliest_start_column, sort_ascending
    )

    # 2. BigM
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_start = min(arrival.values())
    max_deadline = max(deadline.values())
    bigM = math.ceil((max_deadline - min_start + sum_proc_time / math.sqrt(len(machines))) / 1000) * 1000
    print(f"BigM: {bigM}")

    # 3. LP-Modell
    prob = pulp.LpProblem("JSSP_SumAbsoluteLateness", pulp.LpMinimize)

    # 4. Variablen
    starts, ends, abs_lateness = build_jssp_variables(
        jobs, all_ops, arrival, var_cat, abs_lateness={"lowBound": 0, "cat": var_cat}
    )

    # 5. Zielfunktion
    prob += pulp.lpSum(abs_lateness[j] for j in range(len(jobs)))

    # 6. Constraints
    define_technological_constraints(prob, jobs, all_ops, starts, ends, abs_lateness, deadline, mode="absolute_lateness")
    add_machine_constraints(prob, all_ops, starts, machines, epsilon, bigM)

    # 7. Solver
    cmd = get_solver_instance(solver, time_limit, solver_args)
    prob.solve(cmd)

    # 8. Ergebnis
    df_schedule = get_schedule_df(jobs, all_ops, starts, df_jssp, df_times, job_column)

    # 9. Lateness-Spalten berechnen
    df_schedule["Lateness"] = df_schedule["End"] - df_schedule["Deadline"]
    df_schedule["Absolute Lateness"] = df_schedule["Lateness"].abs()

    # 10. Logging
    print("\nSolver-Informationen:")
    print(f"  Summe absolute Lateness  : {round(pulp.value(prob.objective), 4)}")
    print(f"  Solver-Status            : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen         : {len(prob.variables())}")
    print(f"  Anzahl Constraints       : {len(prob.constraints)}")
    print(f"  Laufzeit                 : ~{time.time() - start_time:.0f} Sekunden")

    return df_schedule


# Min. Max Absolute Latenesss -----------------------------------------------------------------------------------------
def solve_jssp_max(df_jssp: pd.DataFrame, df_times: pd.DataFrame,
                                     job_column: str = "Job", earliest_start_column: str = "Arrival",
                                     solver: str = "HiGHS", epsilon: float = 0.0, var_cat: str = "Continuous",
                                     time_limit: int | None = 10800, sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert die maximale absolute Lateness über alle Jobs: min max_j |C_j - d_j|
    """
    start_time = time.time()

    # 1. Vorbereitung
    jobs, all_ops, machines, arrival, deadline = prepare_jssp_inputs(
        df_jssp, df_times, job_column, earliest_start_column, sort_ascending
    )

    # 2. BigM
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_start = min(arrival.values())
    max_deadline = max(deadline.values())
    bigM = math.ceil((max_deadline - min_start + sum_proc_time / math.sqrt(len(machines))) / 1000) * 1000
    print(f"BigM: {bigM}")

    # 3. Modell
    prob = pulp.LpProblem("JSSP_MaxAbsLateness", pulp.LpMinimize)

    # 4. Variablen
    starts, ends, abs_lateness = build_jssp_variables(
        jobs, all_ops, arrival, var_cat, abs_lateness={"lowBound": 0, "cat": var_cat}
    )
    max_abs = pulp.LpVariable("max_abs_lateness", lowBound=0, cat=var_cat)

    # 5. Zielfunktion
    prob += max_abs

    # 6. Constraints
    define_technological_constraints(prob, jobs, all_ops, starts, ends, abs_lateness, deadline, mode="absolute_lateness")
    for j in range(len(jobs)):
        prob += max_abs >= abs_lateness[j]

    add_machine_constraints(prob, all_ops, starts, machines, epsilon, bigM)

    # 7. Solver
    cmd = get_solver_instance(solver, time_limit, solver_args)
    prob.solve(cmd)

    # 8. Ergebnis
    df_schedule = get_schedule_df(jobs, all_ops, starts, df_jssp, df_times, job_column)
    df_schedule["Lateness"] = (df_schedule["End"] - df_schedule["Deadline"]).clip(lower=0).round(2)
    df_schedule["Absolute Lateness"] = df_schedule["Lateness"].abs()

    # 9. Logging
    print("\nSolver-Informationen:")
    print(f"  Maximale absolute Lateness : {round(pulp.value(prob.objective), 4)}")
    print(f"  Solver-Status              : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen           : {len(prob.variables())}")
    print(f"  Anzahl Constraints         : {len(prob.constraints)}")
    print(f"  Laufzeit                   : ~{time.time() - start_time:.0f} Sekunden")

    return df_schedule
