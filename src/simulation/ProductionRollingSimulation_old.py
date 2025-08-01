from src.simulation.sim_utils import duration_log_normal,get_duration, get_time_str
from src.simulation.Machine import Machine
import time
import simpy
import pandas as pd


class ProductionSimulation:
    def __init__(
            self, shift_length: int = 1440, job_column: str = 'Job',
            earliest_start_column: str | None = None, verbose=True, sigma: float =0.2):
        self.job_column = job_column
        self.earliest_start_column = earliest_start_column
        self.verbose = verbose
        self.sigma = sigma

        self.machines = {}
        self.start_time = 0
        self.shift_length = shift_length
        self.pause_time = 0

        self.active_operations = {} # (job_id, operation) → dict mit Op-Daten
        self.finished_log = {}      # (job_id, operation) → Dict mit Op-Daten
        self.all_finished_log = {}  # (job_id, operation) → Dict mit Op-Daten

        self.controller = None
        self.env = None

    def reload_machines(self):
        for name, old_machine in self.machines.items():
            self.machines[name] = Machine(self.env, name) # new SimPy environment

    def add_new_machines(self, dframe_schedule):
        for m in dframe_schedule["Machine"].astype(str).unique():
            if m not in self.machines:
                self.machines[m] = Machine(self.env, m)

    def job_process(self, job_id, job_operations):
        if self.earliest_start_column is not None:
            earliest_start_time = job_operations[0][self.earliest_start_column]
            delay = max(earliest_start_time - self.env.now, 0)
            yield self.env.timeout(delay)

        for op in job_operations:
            machine = self.machines[op["Machine"]]
            planned_duration = op["Processing Time"]

            planned_start = op["Start"] if "Start" in op else self.start_time
            delay = max(planned_start - self.env.now, 0)
            yield self.env.timeout(delay)

            with machine.request() as req:
                yield req
                sim_start = self.env.now
                self.job_started_on_machine(sim_start, job_id, machine)

                sim_duration = duration_log_normal(planned_duration, sigma=self.sigma)
                self.register_active_operation(
                    job_op = op, sim_start = sim_start,
                    planned_duration= planned_duration, sim_duration = sim_duration,
                )

                yield self.env.timeout(sim_duration)
                sim_end = self.env.now
                self.job_finished_on_machine(sim_end, job_id, machine, sim_duration)

            self.register_finished_operation(
                job_op=op, sim_start=sim_start,
                sim_end=sim_end, sim_duration=sim_duration
            )

    def resume_operation_process(self, job_id, op):
        remaining_time = max(0, op["End"] - self.start_time)

        machine = self.machines[op["Machine"]]
        if self.verbose:
            print(f"[{get_time_str(self.env.now)}] Job {job_id}, Operation {op['Operation']} resumed on {op["Machine"]}"
                  + f" (with {get_duration(remaining_time)} left)")

        with machine.request() as req:
            yield req
            yield self.env.timeout(remaining_time)
            sim_end = self.env.now

            self.job_finished_on_machine(sim_end, job_id, machine, remaining_time)

            sim_start = op["Start"]
            self.register_finished_operation(
                job_op=op,
                sim_start=sim_start,
                sim_end=sim_end,
                sim_duration=sim_end - sim_start
            )

    def run(self, dframe_schedule_plan: pd.DataFrame | None  = None , start_time: int = 0, end_time: int | None = None):
        self.start_time = start_time
        self.pause_time = end_time
        self.env = simpy.Environment(initial_time=start_time)
        self.reload_machines()
        self.finished_log = {}

        for (job_id, operation_id), op in self.active_operations.items():
            self.env.process(self.resume_operation_process(job_id, op))

        if dframe_schedule_plan is not None:
            self.add_new_machines(dframe_schedule_plan)

            jobs_grouped = dframe_schedule_plan.groupby(self.job_column)
            for job_id, group in jobs_grouped:
                operations = group.sort_values("Operation").to_dict("records")
                self.env.process(self.job_process(job_id, operations))

        if end_time is not None:
            self.env.run(until=end_time)
        else:
            self.env.run()

    def initialize_run(self, dframe_schedule_plan: pd.DataFrame, start_time: int = 0):
        end_time = start_time + self.shift_length
        self.run(dframe_schedule_plan=dframe_schedule_plan, start_time=start_time, end_time=end_time)

    def continue_run(self, dframe_schedule_plan: pd.DataFrame):
        if self.pause_time is None:
            raise ValueError("Simulation must be initialized before continuing.")

        start_time = self.pause_time
        end_time = start_time + self.shift_length
        self.run(dframe_schedule_plan, start_time=start_time, end_time=end_time)

    def job_started_on_machine(self, time_stamp, job_id, machine):
        if self.verbose:
            print(f"[{get_time_str(time_stamp)}] Job {job_id} started on {machine.name}")
        if self.controller:
            self.controller.job_started_on_machine(time_stamp, job_id, machine)
            time.sleep(0.05)

    def job_finished_on_machine(self, time_stamp, job_id, machine, sim_duration):
        if self.verbose:
            print(f"[{get_time_str(time_stamp)}] Job {job_id} finished on {machine.name} (after {get_duration(sim_duration)})")
        if self.controller:
            self.controller.job_finished_on_machine(time_stamp, job_id, machine, sim_duration)
            time.sleep(0.14)

    def register_active_operation(self, job_op, sim_start, planned_duration, sim_duration):
        key = (job_op[self.job_column], job_op["Operation"])
        entry = {
            self.job_column: job_op[self.job_column],
            "Operation": job_op["Operation"],
            "Machine": job_op["Machine"],
            "Start": sim_start,
            "Processing Time": sim_duration,
            "End": sim_start + sim_duration,
            "Expected End": sim_start + planned_duration
        }
        self.active_operations[key] = entry

    def register_finished_operation(self, job_op, sim_start, sim_end, sim_duration):
        entry = {
            self.job_column: job_op[self.job_column],
            "Operation": job_op["Operation"],
            "Machine": job_op["Machine"],
            "Start": round(sim_start, 2),
            "Processing Time": sim_duration,
            "End": round(sim_end, 2),
        }

        self.all_finished_log[(job_op[self.job_column], job_op["Operation"])] = entry
        self.finished_log[(job_op[self.job_column], job_op["Operation"])] = entry

        self.active_operations.pop((job_op[self.job_column], job_op["Operation"]), None)

    def get_active_operations(self):
        return self.active_operations

    def set_active_operations(self, active_operations):
        self.active_operations = active_operations

    def set_active_operations_from_df(self, df_active: pd.DataFrame):
        """
        Setzt self.active_operations aus einem DataFrame neu zusammen.
        Die Keys sind Tupel aus (job_id, operation), basierend auf den Spalten
        self.job_column und 'Operation'.
        """
        self.active_operations = {
            (row[self.job_column], row["Operation"]): row.to_dict()
            for _, row in df_active.iterrows()
        }

    def get_finished_operations(self):
        return self.finished_log

    def get_active_operations_df(self):
        if not self.active_operations:
            return None
        df = pd.DataFrame(self.active_operations.values())
        return df.sort_values(by=[self.job_column, "Operation"]).reset_index(drop=True)

    def get_finished_operations_df(self):
        if not self.finished_log:
            return None
        df = pd.DataFrame(self.finished_log.values())
        return df.sort_values(by=[self.job_column, "Operation"]).reset_index(drop=True)

    def get_not_started_operations_df(self, df_schedule_plan):
        # Kombiniere Keys aus aktiven und fertigen Operationen
        started_or_finished = set(self.active_operations.keys()).union(self.finished_log.keys())

        # Filtere alle Zeilen heraus, deren (Job, Operation) NICHT in den gestarteten/fertigen enthalten ist
        df = df_schedule_plan[
            ~df_schedule_plan.apply(
                lambda row: (row[self.job_column], row["Operation"]) in started_or_finished,
                axis=1
            )
        ]
        if df.empty:
            return None
        return df.sort_values(by=[self.job_column, "Operation"]).reset_index(drop=True)


if __name__ == "__main__":

    from configs.path_manager import get_path

    basic_data_path = get_path("data", "examples")
    df_schedule = pd.read_csv(basic_data_path / "lateness_schedule_day_01.csv")

    print("Maschinenbelegungsplan:")
    print(df_schedule.head(5))
    print("\n", "---" * 60)

    print("Simulation:")
    simulation = ProductionSimulation(sigma=0.45)
    simulation.run(df_schedule, start_time=1440, end_time=2880)
    df_execution = simulation.get_finished_operations_df()
    print("\n", df_execution.head(5))
    print("\n", "---" * 60)

    print("Active Operations:")
    df_active = simulation.get_active_operations_df()
    print(df_active)


    print("Not started operations:")
    df_not_started = simulation.get_not_started_operations_df(df_schedule)
    print(df_not_started.head(5))
    print("\n", "---" * 60)
    print("Simulation (nur für aktive Operations):")
    simulation.run(None, start_time=2880, end_time=None)
    print("\n", "---" * 20)
    print(simulation.get_finished_operations_df())

    print(simulation.get_active_operations_df())

    print(f"\nScheduleOperations count: {len(df_schedule)}")
    print(f"Finished Operations count: {len(df_execution)}")
    print(f"Active operations count: {len(df_active)}")
    print(f"Waiting operations count: {len(df_not_started)}")


