import math
import random
import time
import simpy
import pandas as pd

from src.simulation.Machine import Machine

# --- Hilfsfunktionen ---

def get_time_str(minutes_in):
    minutes_total = int(minutes_in)
    seconds = int((minutes_in - minutes_total) * 60)
    hours = minutes_total // 60
    minutes = minutes_total % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def get_duration(minutes_in):
    minutes = int(minutes_in)
    seconds = int(round((minutes_in - minutes) * 60))
    parts = []
    if minutes:
        parts.append(f"{minutes:02} minute{'s' if minutes != 1 else ''}")
    if seconds:
        parts.append(f"{seconds:02} second{'s' if seconds != 1 else ''}")
    return " ".join(parts) if parts else ""

def duration_log_normal(duration, vc=0.2):
    sigma = vc
    mu = math.log(duration)
    result = random.lognormvariate(mu, sigma)
    return round(result, 2)


# --- Simulationsklasse ---

class ProductionDaySimulation:
    def __init__(self, dframe_schedule_plan, job_column: str ='Job', vc=0.2):
        self.vc = vc
        self.dframe_schedule_plan = dframe_schedule_plan

        self.job_column = job_column
        self.jobs = self._init_jobs()
        self.machines = None

        self.start_time = 0
        self.end_time = 1440
        self.starting_times_dict = {}
        self.finished_log = []

        self.controller = None

        self.env = None
        self.stop_event = None


    def _init_machines(self):
        unique_machines = self.dframe_schedule_plan["Machine"].unique()
        return {m: Machine(self.env, m) for m in unique_machines}

    def _init_jobs(self):
        return self.dframe_schedule_plan.groupby(self.job_column)

    def job_process(self, job_id, job_operations):
        for op in job_operations:
            machine = self.machines[op["Machine"]]
            planned_start = op["Start"]
            planned_duration = op["Processing Time"]
            op_id = op["Operation"]

            sim_duration = duration_log_normal(planned_duration, vc=self.vc)
            delay = max(planned_start - self.env.now, 0)
            yield self.env.timeout(delay)

            with machine.request() as req:
                yield req
                sim_start = self.env.now
                if sim_start + (planned_duration/10) >= self.end_time:
                    self.check_and_finish_simulation()
                    return

                self.job_started_on_machine(sim_start, job_id, machine)
                self.starting_times_dict[(job_id, machine.name)] = round(sim_start, 2)

                yield self.env.timeout(sim_duration)
                sim_end = self.env.now

                self.job_finished_on_machine(sim_end, job_id, machine, sim_duration)

            entry = {
                self.job_column: job_id,
            }
            if "Production_Plan_ID" in op:
                entry["Production_Plan_ID"] = op["Production_Plan_ID"]

            entry.update({
                "Operation": op_id,
                "Machine": machine.name,
                "Arrival": op["Arrival"],
                "Start": round(sim_start, 2),
                "Processing Time": sim_duration,
                "End": round(sim_end, 2)
            })
            self.finished_log.append(entry)

            del self.starting_times_dict[(job_id, machine.name)]
            self.check_and_finish_simulation()


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



    def run(self, start_time=0, end_time=1440):
        self.start_time = start_time
        self.end_time = end_time
        self.env = simpy.Environment(initial_time=start_time)
        self.machines = self._init_machines()

        for job_id, group in self.jobs:
            operations = group.sort_values("Start").to_dict("records")
            self.env.process(self.job_process(job_id, operations))
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

    def check_and_finish_simulation(self):
        if self.env.now >= self.end_time and not self.starting_times_dict:
            if not self.stop_event.triggered:
                print(f"\n[{get_time_str(self.env.now)}] Simulation ended! There are no more active Operations")
                self.stop_event.succeed()

    def set_controller(self, controller):
        self.controller = controller
        self.controller.add_machines(*self.machines.values())
        job_ids = sorted(self.dframe_schedule_plan[self.job_column].unique())
        self.controller.update_jobs(*job_ids)
