from dataclasses import dataclass, astuple
from collections import UserDict
from typing import List, Optional, Dict
import pandas as pd

# -------------------------
# 1. RoutingOperation
# -------------------------
@dataclass
class RoutingOperation:
    sequence_number: int
    machine: str
    duration: int

# -------------------------
# 2. RoutingDefinition
# -------------------------
class RoutingDefinition(UserDict):
    """
    Mapping von routing_id zu Liste von RoutingOperationen (nach sequence_number sortiert).
    """
    def add_operation(self, routing_id: str, sequence_number: int, machine: str, duration: int):
        if routing_id not in self:
            self[routing_id] = []
        self[routing_id].append(RoutingOperation(sequence_number, machine, duration))

    def sort_operations(self):
        for routing_id in self:
            self[routing_id].sort(key=lambda op: op.sequence_number)

    def get_operation(self, routing_id: str, sequence_number: int) -> Optional[RoutingOperation]:
        return next((op for op in self[routing_id] if op.sequence_number == sequence_number), None)

# -------------------------
# 3. JobOperation
# -------------------------
@dataclass
class JobOperation:
    job_id: str
    routing_id: str
    sequence_number: int

# -------------------------
# 4. JobWorkflowOperation
# -------------------------
@dataclass
class JobWorkflowOperation:
    sequence_number: int
    start_time: int
    end_time: int

# -------------------------
# 5. JobOperationProblemCollection
# -------------------------
class JobOperationProblemCollection:
    """
    Enthält die Liste aller JobOperation-Instanzen und kennt die RoutingDefinitionen.
    """
    def __init__(self, routing_definitions: RoutingDefinition):
        self.routing_definitions = routing_definitions
        self.job_operations: List[JobOperation] = []

    def add_job_operation(self, job_id: str, routing_id: str, sequence_number: int):
        self.job_operations.append(JobOperation(job_id, routing_id, sequence_number))

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for job_op in self.job_operations:
            routing_op = self.routing_definitions.get_operation(job_op.routing_id, job_op.sequence_number)
            if routing_op:
                rows.append({
                    "Job": job_op.job_id,
                    "Routing": job_op.routing_id,
                    "Operation": job_op.sequence_number,
                    "Machine": routing_op.machine,
                    "Processing Time": routing_op.duration
                })
        return pd.DataFrame(rows)

    def get_full_operation(self, job_id: str, routing_id: str, sequence_number: int) -> Optional[tuple]:
        routing_op = self.routing_definitions.get_operation(routing_id, sequence_number)
        if routing_op:
            return (job_id, routing_id, sequence_number, routing_op.machine, routing_op.duration)
        return None

# -------------------------
# 6. JobOperationWorkflowCollection
# -------------------------
class JobOperationWorkflowCollection:
    """
    Erweiterung um konkrete Start- und Endzeiten pro Job und Operation.
    """
    def __init__(self, problem: JobOperationProblemCollection):
        self.problem = problem
        self.workflow_data: Dict[str, List[JobWorkflowOperation]] = {}

    def add_workflow_operation(self, job_id: str, sequence_number: int, start_time: int, end_time: int):
        if job_id not in self.workflow_data:
            self.workflow_data[job_id] = []
        self.workflow_data[job_id].append(JobWorkflowOperation(sequence_number, start_time, end_time))

    def to_dataframe(self) -> pd.DataFrame:
        base_df = self.problem.to_dataframe()
        wf_rows = []
        for job_id, wf_ops in self.workflow_data.items():
            for wf_op in wf_ops:
                wf_rows.append({
                    "Job": job_id,
                    "Operation": wf_op.sequence_number,
                    "Start": wf_op.start_time,
                    "End": wf_op.end_time
                })
        wf_df = pd.DataFrame(wf_rows)
        return base_df.merge(wf_df, on=["Job", "Operation"], how="left")


# -------------------------
# 7. JobInformation
# -------------------------
@dataclass
class JobInformation:
    earliest_start: int
    deadline: int

# -------------------------
# 8. JobInformationCollection
# -------------------------
class JobInformationCollection(UserDict):
    def add_job(self, job_id: str, earliest_start: int, deadline: int):
        self[job_id] = JobInformation(earliest_start, deadline)

    def get_earliest_start(self, job_id: str) -> int:
        return self[job_id].earliest_start

    def get_deadline(self, job_id: str) -> int:
        return self[job_id].deadline

    def get_jobs_by_earliest_start(self, time_point: int) -> dict:
        return {
            job_id: info
            for job_id, info in self.items()
            if info.earliest_start == time_point
        }

    def get_dict(self) -> dict:
        return {
            job_id: astuple(info)
            for job_id, info in self.items()
        }

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        job_column: str = "Job",
        earliest_start_column: str = "Ready Time",
        deadline_column: str = "Deadline"
    ):
        obj = cls()
        for _, row in df.iterrows():
            job_id = row[job_column]
            earliest_start = int(row[earliest_start_column])
            deadline = int(row[deadline_column])
            obj.add_job(job_id, earliest_start, deadline)
        return obj

# -------------------------
# 9. Testcode
# -------------------------
if __name__ == "__main__":
    # 1. RoutingDefinition erzeugen
    routing_def = RoutingDefinition()
    routing_def.add_operation("R25", 0, "M1", 5)
    routing_def.add_operation("R25", 1, "M2", 3)
    routing_def.add_operation("R25", 2, "M3", 4)
    routing_def.add_operation("R30", 0, "M1", 2)
    routing_def.add_operation("R30", 1, "M3", 6)
    routing_def.sort_operations()

    # 2. JobOperationProblemCollection mit Jobs und Routings
    job_problem = JobOperationProblemCollection(routing_def)
    job_problem.add_job_operation("Job1", "R25", 0)
    job_problem.add_job_operation("Job1", "R25", 1)
    job_problem.add_job_operation("Job2", "R30", 0)
    job_problem.add_job_operation("Job2", "R30", 1)

    # 3. JobOperationWorkflowCollection mit Zeiten
    job_workflow = JobOperationWorkflowCollection(job_problem)
    job_workflow.add_workflow_operation("Job1", 0, 0, 5)
    job_workflow.add_workflow_operation("Job1", 1, 6, 9)
    job_workflow.add_workflow_operation("Job2", 0, 0, 2)
    job_workflow.add_workflow_operation("Job2", 1, 3, 9)

    # 4. Ausgabe
    print("\nWorkflow DataFrame:")
    print(job_workflow.to_dataframe())

    print("\nEinzelner Zugriff:")
    print(job_problem.get_full_operation("Job1", "R25", 1))



    print("\n--- JobInformationCollection ---")
    jobinfo = JobInformationCollection()
    jobinfo.add_job("Job1", earliest_start=0, deadline=12)
    jobinfo.add_job("Job2", earliest_start=0, deadline=10)

    print("Deadline von Job2:", jobinfo.get_deadline("Job2"))
    print("Jobs mit earliest_start = 0:", list(jobinfo.get_jobs_by_earliest_start(0).keys()))