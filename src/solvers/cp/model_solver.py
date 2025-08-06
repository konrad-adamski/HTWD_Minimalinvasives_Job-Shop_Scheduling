import contextlib
import os
import sys
from typing import List, Tuple, Dict, Optional, Any

from ortools.sat.python import cp_model

from src.classes.Collection import JobMixCollection
from src.classes.orm_models import JobOperation
from src.solvers.cp.model_classes import OperationIndexMapper


def solve_cp_model_and_extract_schedule(
        model: cp_model.CpModel, index_mapper: OperationIndexMapper,
        starts: Dict[Tuple[int, int], cp_model.IntVar], ends: Dict[Tuple[int, int], cp_model.IntVar],
        msg: bool, time_limit: Optional[int] = None, gap_limit: float = 0.0,
        log_file: Optional[str] = None) -> Tuple[JobMixCollection, Dict[str, Any]]:
    """
    Solves a CP-SAT model and extracts the schedule if feasible.

    :param model: The CP-SAT model to solve.
    :param operations: List of operation metadata tuples:
                       (job_idx, job_name, op_idx, op_id, machine, duration).
    :param starts: Mapping from (job_idx, op_idx) to start time variable.
    :param ends: Mapping from (job_idx, op_idx) to end time variable.
    :param msg: Whether to enable solver log output.
    :param time_limit: Optional maximum time for solver (in seconds).
    :param gap_limit: Acceptable relative gap limit.
    :param log_file: Optional path to file for redirecting solver output.
    :return: Tuple of (scheduled operations list, solver info dict).
    """
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    if time_limit is not None:
        solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.relative_gap_limit = gap_limit

    if log_file is not None:
        with _redirect_cpp_logs(log_file):
            status = solver.Solve(model)
    else:
        status = solver.Solve(model)

    solver_info = {
        "status": solver.StatusName(status),
        "objective_value": solver.ObjectiveValue() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None,
        "best_objective_bound": solver.BestObjectiveBound(),
        "number_of_branches": solver.NumBranches(),
        "wall_time": solver.WallTime()
    }

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return _extract_cp_schedule_from_operations(index_mapper, starts, ends, solver), solver_info

    return JobMixCollection(), solver_info


def _extract_cp_schedule_from_operations(
        index_mapper: OperationIndexMapper, starts: Dict[Tuple[int, int], cp_model.IntVar],
        ends: Dict[Tuple[int, int], cp_model.IntVar], solver: cp_model.CpSolver) -> JobMixCollection:
    """
    Extracts the final schedule based on flattened operations and CP variables.

    :param operations: List of operations in the form (job_idx, job, op_idx, op_id, machine, duration).
    :type operations: List[Tuple[int, str, int, int, str, int]]
    :param starts: Dictionary of start time variables indexed by (job_idx, op_idx).
    :type starts: Dict[Tuple[int, int], cp_model.IntVar]
    :param ends: Dictionary of end time variables indexed by (job_idx, op_idx).
    :type ends: Dict[Tuple[int, int], cp_model.IntVar]
    :param solver: The CP-SAT solver instance after solving the model.
    :type solver: cp_model.CpSolver

    :return: List of (job, op_id, machine, start_time, duration, end_time) tuples.
    :rtype: List[Tuple[str, int, str, int, int, int]]
    """

    schedule_job_collection = JobMixCollection()

    for (job_idx, op_idx), operation in index_mapper.items():
        start = solver.Value(starts[(job_idx, op_idx)])
        end = solver.Value(ends[(job_idx, op_idx)])

        schedule_job_collection.add_operation_instance(
            op = operation,
            new_start=start,
            new_end=end
        )

    return schedule_job_collection



@contextlib.contextmanager
def _redirect_cpp_logs(logfile_path: str = "cp_output.log"):
    """
    Context manager to temporarily redirect stdout/stderr,
    e.g. to capture output from OR-Tools CP-SAT solver or other C++ logs.
    After the block, original output streams are restored.
    """

    # Flush any current output to avoid mixing content
    sys.stdout.flush()
    sys.stderr.flush()

    # Save original file descriptors for stdout and stderr
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)

    with open(logfile_path, 'w') as f:
        try:
            # Redirect stdout and stderr to the log file
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
            yield
            f.flush()  # Ensures content is flushed to file, esp. in Jupyter
        finally:
            # Restore original stdout and stderr
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)