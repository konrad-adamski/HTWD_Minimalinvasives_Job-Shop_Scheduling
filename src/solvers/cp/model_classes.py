import math
from dataclasses import dataclass
from collections import UserDict
from typing import Optional, Union, Dict, Tuple

from src.classes.orm_models import JobTemplate, Job, JobOperation


@dataclass
class MachineFixInterval:
    machine: str
    start: int
    end: int

class MachineFixIntervalMap(UserDict):
    def add_interval(self, machine: str, start: int, end: float):
        """Set or replace fix interval information for a machine."""
        self.data[machine] = MachineFixInterval(machine, start, int(math.ceil(end)))

    def update_interval(self, machine: str, end: float):
        """Only updates the fix interval end time if it is larger than the current."""
        current = self.data.get(machine)
        if current is None or end > current.end:
            start = current.start if current else 0
            self.data[machine] = MachineFixInterval(machine, start, int(math.ceil(end)))

    def get_interval(self, machine: str) -> Optional[MachineFixInterval]:
        return self.data.get(machine)



@dataclass
class JobDelay:
    job_id: str
    earliest_start: int

class JobDelayMap(UserDict):
    def add_delay(self, job_id: str, time_stamp: float):
        """Set or replace delay information for a job."""
        self.data[job_id] = JobDelay(job_id, int(math.ceil(time_stamp)))

    def update_delay(self, job_id: str, time_stamp: float):
        """Only updates the time_stamp if it is larger than the current."""
        current = self.data.get(job_id)
        if current is None or time_stamp > current.time_stamp:
            self.data[job_id] = JobDelay(job_id, int(math.ceil(time_stamp)))

    def get_delay(self, job_id: str) -> Optional[JobDelay]:
        return self.data.get(job_id)


class OperationIndexMapper:
    def __init__(self):
        self.index_to_operation: Dict[Tuple[int, int], JobOperation] = {}

    def add(self, job_idx: int, op_idx: int, operation: JobOperation):
        self.index_to_operation[(job_idx, op_idx)] = operation

    def items(self):
        return self.index_to_operation.items()

    def keys(self):
        return self.index_to_operation.keys()

    def values(self):
        return self.index_to_operation.values()

    def get_index_from_operation(self, operation: JobOperation) -> Optional[Tuple[int, int]]:
        for index, op in self.index_to_operation.items():
            if op == operation:
                return index
        return None  # Falls nicht gefunden