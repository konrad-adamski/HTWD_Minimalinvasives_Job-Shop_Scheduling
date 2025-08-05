import math
from dataclasses import dataclass
from collections import UserDict
from typing import Optional, Union

from src.classes.orm_models import JobTemplate, Job


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
    job: Union[Job, JobTemplate]
    earliest_start: int

class JobDelayMap(UserDict):
    def add_delay(self, job: Union[Job, JobTemplate], time_stamp: float):
        """Set or replace delay information for a job."""
        self.data[job] = JobDelay(job, int(math.ceil(time_stamp)))

    def update_delay(self, job: Union[Job, JobTemplate], time_stamp: float):
        """Only updates the time_stamp if it is larger than the current."""
        current = self.data.get(job)
        if current is None or time_stamp > current.time_stamp:
            self.data[job] = JobDelay(job, int(math.ceil(time_stamp)))

    def get_delay(self, job: Union[Job, JobTemplate]) -> Optional[JobDelay]:
        return self.data.get(job)