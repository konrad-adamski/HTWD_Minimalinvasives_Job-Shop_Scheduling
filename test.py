from ortools.sat.python import cp_model

def sum_values(solver: cp_model.CpSolver, vars_):
    return sum(solver.Value(v) for v in vars_) if vars_ else 0

def get_cost_breakdown(solver: cp_model.CpSolver,
                       w_t: int, w_e: int, w_first: int, w_dev: int,
                       tardiness_vars, earliness_vars, first_op_vars, deviation_vars):
    tardiness_sum = sum_values(solver, tardiness_vars)
    earliness_sum = sum_values(solver, earliness_vars)
    first_sum     = sum_values(solver, first_op_vars)
    dev_sum       = sum_values(solver, deviation_vars)

    costs = {
        "tardiness_raw": tardiness_sum,
        "earliness_raw": earliness_sum,
        "first_raw":     first_sum,
        "deviation_raw": dev_sum,
        "tardiness_cost": w_t * tardiness_sum,
        "earliness_cost": w_e * earliness_sum,
        "first_cost":     w_first * first_sum,
        "deviation_cost": w_dev * dev_sum,
    }
    costs["total_cost_estimated"] = (
        costs["tardiness_cost"] + costs["earliness_cost"] +
        costs["first_cost"] + costs["deviation_cost"]
    )
    return costs
