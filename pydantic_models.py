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
# 2. RoutingOperationCollection
# -------------------------
class RoutingOperationCollection(UserDict):
    """
    Sammlung von RoutingOperationen pro routing_id (nach sequence_number sortiert).
    """
    def add_operation(self, routing_id: str, sequence_number: int, machine: str, duration: int):
        # Falls das Routing noch nicht existiert, initialisiere leere Liste
        if routing_id not in self:
            self[routing_id] = []

        # Suche nach bestehender Operation mit gleicher sequence_number und überschreibe sie
        for i, op in enumerate(self[routing_id]):
            if op.sequence_number == sequence_number:
                self[routing_id][i] = RoutingOperation(sequence_number, machine, duration)
                return  # Überschreiben abgeschlossen, keine weitere Aktion notwendig

        # Falls keine passende Operation existiert, neu hinzufügen
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
    job_id: str
    routing_id: str
    sequence_number: int
    start_time: int
    end_time: int

# -------------------------
# 5. JobOperationProblemCollection
# -------------------------
class JobOperationProblemCollection:
    """
    Enthält die Liste aller JobOperation-Instanzen und kennt die RoutingOperationen.
    """
    def __init__(self, routing_definitions: RoutingOperationCollection):
        self.routing_definitions = routing_definitions
        self.job_operations: List[JobOperation] = []

    def add_job_operation(self, job_id: str, routing_id: str, sequence_number: int):
        # Falls bereits vorhanden, nichts tun
        if any(op.job_id == job_id and op.sequence_number == sequence_number for op in self.job_operations):
            return
        # Andernfalls hinzufügen
        self.job_operations.append(JobOperation(job_id, routing_id, sequence_number))

    def get_full_operation(self, job_id: str, routing_id: str, sequence_number: int) -> Optional[tuple]:
        routing_op = self.routing_definitions.get_operation(routing_id, sequence_number)
        if routing_op:
            return job_id, routing_id, sequence_number, routing_op.machine, routing_op.duration
        return None

    def to_dataframe(
            self, job_column: str = "Job", routing_column: str = "Routing", operation_column: str = "Operation",
            machine_column: str = "Machine", duration_column: str = "Processing Time") -> pd.DataFrame:
        rows = []
        for job_op in self.job_operations:
            if job_op.routing_id not in self.routings_collection:
                print(f"[Warnung] Routing '{job_op.routing_id}' fehlt – Zeile wird übersprungen!")
                continue
            routing_op = self.routings_collection.get_operation(job_op.routing_id, job_op.sequence_number)
            if routing_op:
                rows.append({
                    job_column: job_op.job_id,
                    routing_column: job_op.routing_id,
                    operation_column: job_op.sequence_number,
                    machine_column: routing_op.machine,
                    duration_column: routing_op.duration
                })
        return pd.DataFrame(rows)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, routing_definitions: RoutingOperationCollection,
                       job_column: str = "Job", routing_column: str = "Routing", operation_column: str = "Operation"):
        obj = cls(routing_definitions)
        for _, row in df.iterrows():
            job_id = str(row[job_column])
            routing_id = str(row[routing_column])
            sequence_number = int(row[operation_column])
            obj.add_job_operation(job_id, routing_id, sequence_number)
        return obj

# -------------------------
# 6. JobOperationWorkflowCollection
# -------------------------
class JobOperationWorkflowCollection:
    """
    Enthält die Liste aller JobWorkflowOperation-Instanzen und kennt die RoutingOperationen.
    """
    def __init__(self, routing_definitions: RoutingOperationCollection):
        self.routing_definitions = routing_definitions
        self.job_operations: List[JobWorkflowOperation] = []

    def add_workflow_operation(self, job_id: str, routing_id: str, sequence_number: int, start_time: int, end_time: int):
        self.job_operations.append(JobWorkflowOperation(job_id, routing_id, sequence_number, start_time, end_time))

    def get_full_operation(self, job_id: str, routing_id: str, sequence_number: int) -> Optional[tuple]:
        for op in self.job_operations:
            if op.job_id == job_id and op.routing_id == routing_id and op.sequence_number == sequence_number:
                routing_op = self.routing_definitions.get_operation(routing_id, sequence_number)
                if routing_op:
                    return job_id, routing_id, sequence_number, routing_op.machine, routing_op.duration, op.start_time, op.end_time
        return None

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for op in self.job_operations:
            routing_op = self.routing_definitions.get_operation(op.routing_id, op.sequence_number)
            if routing_op:
                rows.append({
                    "Job": op.job_id,
                    "Routing": op.routing_id,
                    "Operation": op.sequence_number,
                    "Machine": routing_op.machine,
                    "Processing Time": routing_op.duration,
                    "Start": op.start_time,
                    "End": op.end_time
                })
        return pd.DataFrame(rows)

# -------------------------
# 7. Testcode
# -------------------------
if __name__ == "__main__":
    # 1. RoutingOperationCollection erzeugen
    routing_def = RoutingOperationCollection()
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
    job_workflow = JobOperationWorkflowCollection(routing_def)
    job_workflow.add_workflow_operation("Job1", "R25", 0, 0, 5)
    job_workflow.add_workflow_operation("Job1", "R25", 1, 6, 9)
    job_workflow.add_workflow_operation("Job2", "R30", 0, 0, 2)
    job_workflow.add_workflow_operation("Job2", "R30", 1, 3, 9)


        # 4. Ausgabe
    print("\nEinzelner Zugriff:")
    print(job_problem.get_full_operation("Job1", "R25", 1))

    # 5. Zusätzliche Tests für RoutingOperationCollection
    print("\n--- Zusätzliche Tests ---")
    print("Alle Operationen in R25 vor Überschreiben:")
    for op in routing_def["R25"]:
        print(op)

    # Test: Überschreiben einer bestehenden Operation
    routing_def.add_operation("R25", 1, "M99", 90)

    print("\nAlle Operationen in R25 nach Überschreiben von sequence_number=1:")
    for op in routing_def["R25"]:
        print(op)

    # Test: Zugriff auf spezifische Operation
    op = routing_def.get_operation("R25", 1)
    print("\nZugriff auf R25, sequence 1:", op)

    print("\nWorkflow DataFrame:")
    print(job_workflow.to_dataframe())

    print("\nEinzelner Zugriff:")
    print(job_problem.get_full_operation("Job1", "R25", 1))
