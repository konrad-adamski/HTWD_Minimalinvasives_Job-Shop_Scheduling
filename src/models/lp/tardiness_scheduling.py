from src.models.lp.solver_builder import *
import pandas as pd
import math
import pulp
import time

def solve_jssp_sum(df_jssp: pd.DataFrame, df_times: pd.DataFrame, job_column: str = "Job", earliest_start_column = "Arrival"
                   solver: str = "HiGHS", epsilon: float = 0.0, var_cat: str = "Continuous",
                   time_limit: int | None = 10800, sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    start_time = time.time()

    # 1. Vorverarbeitung & Variablen
    jobs, all_ops, machines, earliest_start, deadline = prepare_jssp_inputs(df_jssp, df_times, job_column, earliest_start_column, sort_ascending)

    starts, ends, tards = build_jssp_variables(jobs, all_ops, earliest_start, var_cat,
                                                          tards={"lowBound": 0, "cat": var_cat})
    
    # 2. Modellaufbau
    prob = pulp.LpProblem("JSSP_SumTardiness", pulp.LpMinimize)
    
    # Zielfunktion: Minimierung der Summe der Tardiness (Verspätung) aller Jobs
    prob += pulp.lpSum(tards.values())

    # 3. Constraints
    define_technological_constraints(prob, jobs, all_ops, starts, ends, tards, deadline, mode="tardiness")

    sum_proc_time = df_jssp["Processing Time"].sum()
    min_earliest_start = min(earliest_start.values())
    max_deadline = max(deadline.values())
    bigM = math.ceil((max_deadline - min_earliest_start + sum_proc_time / math.sqrt(len(machines))) / 1000) * 1000

    add_machine_constraints(prob, all_ops, starts, machines, epsilon, bigM)

    # 4. Lösen
    solver_instance = get_solver_instance(solver, time_limit, solver_args)
    prob.solve(solver_instance)
    objective_value = pulp.value(prob.objective)

    # 5. Ergebnis
    df_schedule = get_records_df(df_jssp, df_times, jobs, starts, job_column=job_column)
    df_schedule["Tardiness"] = (df_schedule["End"] - df_schedule["Deadline"]).clip(lower=0).round(2)
    df_schedule = df_schedule.sort_values([job_column, "Operation"]).reset_index(drop=True)

    # 6. Logging
    print("\nSolver-Informationen:")
    print(f"  Summe Tardiness         : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")
    print(f"  Laufzeit                : ~{time.time() - start_time:.0f} Sekunden")

    return df_schedule


def solve_jssp_max(df_jssp: pd.DataFrame, df_times: pd.DataFrame, job_column: str = "Job", earliest_start_column: str = "Arrival",
                   solver: str = "HiGHS", epsilon: float = 0.0, var_cat: str = "Continuous",
                   time_limit: int | None = 10800, sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    start_time = time.time()

    # 1. Vorverarbeitung & Variablen
    jobs, all_ops, machines, earliest_start, deadline = prepare_jssp_inputs(
        df_jssp, df_times, job_column, earliest_start_column, sort_ascending
    )
    starts, ends, tards = build_jssp_variables(jobs, all_ops, earliest_start, var_cat,
                                                      tards={"lowBound": 0, "cat": var_cat})

    # 2. Modellaufbau
    prob = pulp.LpProblem("JSSP_MaxTardiness", pulp.LpMinimize)

    # Zielfunktion: Minimierung der maximalen Tardiness
    max_tard = pulp.LpVariable("max_tardiness", lowBound=0, cat=var_cat)
    prob += max_tard

    # 3. Constraints
    define_technological_constraints(prob, jobs, all_ops, starts, ends, tards, deadline, mode="tardiness")

    # max_tard ≥ Tardiness[j]
    for j in range(len(jobs)):
        prob += max_tard >= tards[j]

    sum_proc_time = df_jssp["Processing Time"].sum()
    min_earliest_start = min(earliest_start.values())
    max_deadline = max(deadline.values())
    bigM = math.ceil((max_deadline - min_earliest_start + sum_proc_time / math.sqrt(len(machines))) / 1000) * 1000

    add_machine_constraints(prob, all_ops, starts, machines, epsilon, bigM)

    # 4. Lösen
    solver_instance = get_solver_instance(solver, time_limit, solver_args)
    prob.solve(solver_instance)
    objective_value = pulp.value(prob.objective)

    # 5. Ergebnis
    df_schedule = get_records_df(df_jssp, df_times, jobs, starts, job_column=job_column)
    df_schedule["Tardiness"] = (df_schedule["End"] - df_schedule["Deadline"]).clip(lower=0).round(2)
    df_schedule = df_schedule.sort_values(["Start", job_column, "Operation"]).reset_index(drop=True)

    # 6. Logging
    print("\nSolver-Informationen:")
    print(f"  Maximale Tardiness      : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")
    print(f"  Laufzeit                : ~{time.time() - start_time:.0f} Sekunden")

    return df_schedule


