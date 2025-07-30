from dataclasses import dataclass
from typing import Optional


# ----- RoutingOperation -----
@dataclass(frozen=True)
class RoutingOperation:
    sequence_number: int
    machine: str
    duration: int

# ----- JobOperation -----
@dataclass(frozen=True)
class JobOperation:
    job_id: str
    routing_id: str
    sequence_number: int

@dataclass
class JobOperationView:
    job_id: str
    routing_id: str
    sequence_number: int
    machine: str
    duration: int







