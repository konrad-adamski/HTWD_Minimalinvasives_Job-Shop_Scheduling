from src.simulation.sim_utils import duration_log_normal,get_duration, get_time_str
from src.simulation.Machine import Machine
import time
import simpy
import pandas as pd


class ProductionSimulation:
    def __init__(self, job_column: str = 'Job', earliest_start_column='Arrival', verbose=True, sigma=0.2):
        self.job_column = job_column
        self.earliest_start_column = earliest_start_column
        self.verbose = verbose
        self.sigma = sigma

        self.machines = {}
        self.start_time = 0
        self.end_time = None

        self.active_operations = {}  # (job_id, operation) → dict mit Op-Daten
        self.finished_log = {}  # (job_id, operation) → Dict mit Op-Daten
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
        remaining_time = op["End"] - self.start_time
        machine = self.machines[op["Machine"]]
        print(f"[{get_time_str(self.env.now)}] Job {job_id}, Operation {op['Operation']} resumed with {remaining_time:.2f} min")

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
        self.end_time = end_time
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
        self.active_operations[(job_op[self.job_column], job_op["Operation"])] = {
            self.job_column: job_op[self.job_column],
            "Operation": job_op["Operation"],
            "Machine": job_op["Machine"],
            self.earliest_start_column: job_op[self.earliest_start_column],
            "Arrival": job_op["Arrival"],
            "Start": sim_start,
            "Planned Duration": planned_duration,
            "Processing Time": sim_duration,
            "Expected End": sim_start + planned_duration,
            "End": sim_start + sim_duration
        }

    def register_finished_operation(self, job_op, sim_start, sim_end, sim_duration):
        entry = {"Routing_ID": job_op["Routing_ID"]} if "Routing_ID" in job_op else {}
        entry.update({
            self.job_column: job_op[self.job_column],
            "Operation": job_op["Operation"],
            "Machine": job_op["Machine"],
            self.earliest_start_column: job_op[self.earliest_start_column],
            "Arrival": job_op["Arrival"],
            "Start": round(sim_start, 2),
            "Processing Time": sim_duration,
            "End": round(sim_end, 2),
        })


        self.all_finished_log[(job_op[self.job_column], job_op["Operation"])] = entry
        self.finished_log[(job_op[self.job_column], job_op["Operation"])] = entry

        if (job_op[self.job_column], job_op["Operation"]) in self.active_operations:
            del self.active_operations[(job_op[self.job_column], job_op["Operation"])]

    def get_active_operations(self):
        return self.active_operations

    def get_finished_operations(self):
        return self.finished_log

    def get_active_operations_df(self):
        df = pd.DataFrame(self.active_operations.values())
        return df.sort_values(by=[self.job_column, "Operation"]).reset_index(drop=True)

    def get_finished_operations_df(self):
        df =pd.DataFrame(self.finished_log.values())
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
        return df.sort_values(by=[self.job_column, "Operation"]).reset_index(drop=True)


    def set_controller(self, controller):
        self.controller = controller
        self.controller.add_machines(*self.machines.values())
        # job_ids = sorted(self.dframe_schedule_plan[self.job_column].unique())
        # self.controller.update_jobs(*job_ids)

if __name__ == "__main__":
    pass
    """
    from configs.path_manager import get_path

    basic_data_path = get_path("data", "basic")
    df_schedule = pd.read_csv(basic_data_path / "schedule_example.csv")

    print("Maschinenbelegungsplan:")
    print(df_schedule.head(5))
    print("\n", "---" * 60)

    print("Simulation:")
    simulation = ProductionSimulation(sigma=0.45)
    simulation.run(df_schedule, start_time=0, end_time=1440 - 1)
    df_execution = simulation.get_finished_operations_df()
    print(df_execution.head(5))
    print("\n", "---" * 60)

    print("Active Operations:")
    print(simulation.get_active_operations_df())


    print("Not started operations:")
    df_not_started = simulation.get_not_started_operations_df(df_schedule)
    print(df_not_started.head(5))
    print("\n", "---" * 60)
    simulation.run(None, start_time=1440, end_time=None)
    print("\n", "---" * 20)
    print(simulation.get_finished_operations_df())

"""
