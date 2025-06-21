from src.simulation.Machine import Machine
from src.simulation.sim_utils import duration_log_normal,get_duration, get_time_str

import time
import simpy
import pandas as pd

# --- Simulationsklasse ---
class ProductionSimulation:
    def __init__(self, dframe_schedule_plan, job_column: str ='Job',earliest_start_column='Arrival', vc=0.2):
        self.vc = vc
        self.dframe_schedule_plan = dframe_schedule_plan

        self.job_column = job_column
        self.earliest_start_column = earliest_start_column
        self.jobs = self._init_jobs()

        self.machines = None
        self.start_time = 0
        self.end_time = None
        self.starting_times_dict = {}
        self.finished_log = []

        self.controller = None
        self.env = None
        self.stop_event = None


    def _init_machines(self):
        unique_machines = self.dframe_schedule_plan["Machine"].unique()
        return {m: Machine(self.env, m) for m in unique_machines}

    def _init_jobs(self):
        df = self.dframe_schedule_plan.copy()
        df = df.sort_values([self.job_column, "Operation"])  # Sortiere technologisch korrekt
        return df.groupby(self.job_column)

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

                if self.end_time is not None and sim_start + (planned_duration * 0.10) >= self.end_time: # 10 % der geplanten Zeit als Schwellwert
                    self.check_stop_event()
                    return

                self.job_started_on_machine(sim_start, job_id, machine)
                self.starting_times_dict[(job_id, machine.name)] = round(sim_start, 2)

                sim_duration = duration_log_normal(planned_duration, vc=self.vc)
                yield self.env.timeout(sim_duration)
                sim_end = self.env.now

                self.job_finished_on_machine(sim_end, job_id, machine, sim_duration)

            entry = {
                self.job_column: job_id,
            }
            if "Production_Plan_ID" in op:
                entry["Production_Plan_ID"] = op["Production_Plan_ID"]

            entry["Operation"] = op["Operation"]
            entry["Machine"] = machine.name
            entry["Arrival"] = op["Arrival"]

            # Zusätzlich earliest_start_column, wenn es nicht Arrival ist
            if self.earliest_start_column != "Arrival":
                entry[self.earliest_start_column] = earliest_start_time

            entry["Start"] = round(sim_start, 2)
            entry["Processing Time"] = sim_duration
            entry["End"] = round(sim_end, 2)

            self.finished_log.append(entry)

            del self.starting_times_dict[(job_id, machine.name)]
            self.check_stop_event()


    def end_trigger_process(self):
        # 1. Warte bis self.end_time erreicht ist
        yield self.env.timeout(max(self.end_time - self.env.now - 0.01, 0))

        # 2. Danach prüfe in regelmäßigen Abständen, ob noch laufende Operationen existieren
        while self.starting_times_dict:
            yield self.env.timeout(1)  # prüfe jede Sim-Minute erneut

        # 3. Wenn keine Operationen mehr aktiv sind, beende die Simulation
        if not self.stop_event.triggered:
            print(f"\n[{get_time_str(self.env.now)}] Simulation ended by fallback after end_time.")
            self.stop_event.succeed()

    def run(self, start_time=0, end_time=None):
        self.start_time = start_time
        self.end_time = end_time
        self.env = simpy.Environment(initial_time=start_time)
        self.machines = self._init_machines()

        for job_id, group in self.jobs:
            operations = group.sort_values("Start").to_dict("records")
            self.env.process(self.job_process(job_id, operations))

        if self.end_time is None:
            self.env.run()
        else:
            self.env.process(self.end_trigger_process())
            self.stop_event = self.env.event()
            self.env.run(until=self.stop_event)

        dframe_execution = pd.DataFrame(self.finished_log)
        return dframe_execution.sort_values(by=[self.job_column, "Operation"]).reset_index(drop=True)

    def job_started_on_machine(self, time_stamp, job_id, machine):
        print(f"[{get_time_str(time_stamp)}] Job {job_id} started on {machine.name}")
        if self.controller:
            self.controller.job_started_on_machine(time_stamp, job_id, machine)
            time.sleep(0.05)

    def job_finished_on_machine(self, time_stamp, job_id, machine, sim_duration):
        print(f"[{get_time_str(time_stamp)}] Job {job_id} finished on {machine.name} (after {get_duration(sim_duration)})")
        if self.controller:
            self.controller.job_finished_on_machine(time_stamp, job_id, machine, sim_duration)
            time.sleep(0.14)

    def check_stop_event(self):
        if self.stop_event and self.env.now >= self.end_time and not self.starting_times_dict:
            if not self.stop_event.triggered:
                print(f"\n[{get_time_str(self.env.now)}] Simulation ended! There are no more active Operations")
                self.stop_event.succeed()

    def set_controller(self, controller):
        self.controller = controller
        self.controller.add_machines(*self.machines.values())
        job_ids = sorted(self.dframe_schedule_plan[self.job_column].unique())
        self.controller.update_jobs(*job_ids)
