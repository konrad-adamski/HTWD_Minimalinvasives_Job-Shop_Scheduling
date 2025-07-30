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
# 3. JobOperationProblem
# -------------------------
@dataclass
class JobOperation:
    job_id: str
    routing_id: str
    sequence_number: int

# -------------------------
# 4. JobOperationProblemCollection
# -------------------------
class JobOperationProblemCollection:
    """
    Enthält die Liste aller JobOperationProblem-Instanzen und kennt die RoutingDefinitionen.
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
# 5. Testcode
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

    # 3. Ausgabe
    print("\nDataFrame-Darstellung:")
    print(job_problem.to_dataframe())

    print("\nEinzelner Zugriff:")
    print(job_problem.get_full_operation("Job1", "R25", 1))

