from collections import defaultdict
from typing import Literal, List, Dict, Optional
from src.domain.Collection import LiveJobCollection
from src.domain.orm_models import JobOperation


class Solver:

    def __init__(self, jobs_collection: LiveJobCollection, schedule_start: int = 0):


        # JobsCollections and information
        self.jobs_collection = jobs_collection
        self.previous_schedule_jobs_collection = LiveJobCollection()
        self.active_jobs_collection = LiveJobCollection()

        self.machines = jobs_collection.get_unique_machine_names()
        self.machine_ready_time: Dict[str, int] = {m: schedule_start for m in self.machines}

        self.schedule_start = schedule_start

        self.total_ops = jobs_collection.count_operations()

        for job in self.jobs_collection.values():
            job.current_operation = job.get_first_operation()
            job.current_operation_earliest_start = max(job.earliest_start, schedule_start)


    def get_last_end_per_machine(self, collection: Optional[LiveJobCollection]) -> Dict[str, int]:
        """
        Liefert je Maschine die größte 'end'-Zeit aus der gegebenen LiveJobCollection.
        Operationen ohne gesetztes 'end' werden ignoriert.
        """
        if collection is None:
            return {}

        last_end: Dict[str, int] = {}
        for job in collection.values():
            for op in job.operations:
                if op.end is None:
                    continue
                m = op.machine_name
                e = int(op.end)
                if m not in last_end or e > last_end[m]:
                    last_end[m] = e
        return last_end

    def set_active_jobs_collection(self, active_jobs_collection: Optional[LiveJobCollection]):
        self.active_jobs_collection = active_jobs_collection

        last_end_per_machine = self.get_last_end_per_machine(self.active_jobs_collection)
        for m, e in last_end_per_machine.items():
            self.machine_ready_time[m] = max(self.machine_ready_time.get(m, self.schedule_start), e)

        for active_job in self.active_jobs_collection.values():
            job = self.jobs_collection.get(active_job.id)
            if job is None:
                continue

            last_op = active_job.get_last_operation()
            if last_op is not None:
                job.current_operation_earliest_start = max(self.schedule_start, last_op.end)


    def set_previous_schedule_jobs_collection(self, previous_schedule_jobs_collection: LiveJobCollection):
        self.previous_schedule_jobs_collection = previous_schedule_jobs_collection

    def select_by_priority(
            self,
            conflict_ops: List[JobOperation],
            rule: Literal["SPT", "FCFS", "MWKR", "EDD", "SLACK", "DEVIATION"] = "SPT"
    ) -> Optional[JobOperation]:
        """
        Wählt aus JobOperation-Objekten gemäß Regel:
        - SPT: kürzeste Bearbeitungszeit
        - FCFS: kleinste Job-Ankunftszeit
        - EDD: früheste Job-Deadline
        Fallbacks: arrival -> 0, due -> +inf, start -> 0, job_total_dur -> +inf
        """
        if not conflict_ops:
            return None

        def _duration(op: JobOperation):
            return op.duration

        def _job_earliest_start(op: JobOperation):
            return op.job_earliest_start

        def _job_arrival(op: JobOperation):
            return op.job_arrival if op.job_arrival is not None else 0

        def _job_due_date(op: JobOperation):
            return op.job_due_date if op.job_due_date is not None else 0

        def _job_total_dur(op: JobOperation):
            return op.job.sum_duration

        def _slack(op: JobOperation):
            return _job_due_date(op) - (op.start + _remaining_work(op))


        def _remaining_work(op: JobOperation):
            # inkl. aktueller Operation (deine Methode gibt inkl. Position zurück)
            return op.job.sum_left_duration(op.position_number)


        def _start_deviation(op: JobOperation):
            prev_op_version = self.previous_schedule_jobs_collection.get_operation(op.job_id, op.position_number)
            if prev_op_version is not None:
                #print(prev_op_version, abs(op.start - prev_op_version.start))
                return abs(op.start - prev_op_version.start)
            else:
                return 100 * (self.schedule_start + 1000)    # big_value (but same for all)

        if rule == "SPT":
            key = lambda x: (_duration(x), _job_arrival(x), _job_earliest_start(x), _job_total_dur(x))
            return min(conflict_ops, key=key)

        elif rule == "FCFS":
            key = lambda x: (_job_arrival(x), _job_earliest_start(x), _duration(x), _job_total_dur(x))
            return min(conflict_ops, key=key)

        elif rule == "EDD":
            key = lambda x: (_job_due_date(x), _job_arrival(x), _job_earliest_start(x), _job_total_dur(x))
            return min(conflict_ops, key=key)

        elif rule == "MWKR":
            # Meiste Restarbeit zuerst
            key = lambda x: (_remaining_work(x), - _job_earliest_start(x), - _duration(x))
            return max(conflict_ops, key=key)

        elif rule == "SLACK":
            # kleinste Slack zuerst; bei Gleichstand: frühester Start, dann SPT
            key = lambda x: (_slack(x), _job_earliest_start(x), _duration(x), x.job_id)
            return min(conflict_ops, key=key)

        elif rule == "DEVIATION":
            # kleinste Abweichung zuerst, bei Gleichstand kleinste Slack, dann frühester Start, dann SPT
            key = lambda x: (_start_deviation(x), _slack(x), _job_earliest_start(x), _duration(x), x.job_id)
            return min(conflict_ops, key=key)

        else:
            raise ValueError("Invalid rule")


    def intervals_overlap(self, operation_a: JobOperation, operation_b: JobOperation) -> bool:
        # echte Überlappung (GT-Konfliktlogik)
        #if not (operation_a.end <= operation_b.start or operation_b.end <= operation_a.start):
        #    print(f"{operation_a.start = }, {operation_a.end = };  {operation_b.start = } {operation_b.end = }")

        return not (operation_a.end <= operation_b.start or operation_b.end <= operation_a.start)


    def get_machine_candidates(self) -> Dict[str, list[JobOperation]]:
        machine_candidates: Dict[str, list[JobOperation]] = defaultdict(list)
        for job in self.jobs_collection.values():
            operation = job.current_operation
            if operation is not None:
                operation.start = max(self.machine_ready_time[operation.machine_name], job.current_operation_earliest_start)
                operation.end = operation.start + operation.duration
                machine_candidates[operation.machine_name].append(operation)
        return dict(machine_candidates)

    def solve(self, priority_rule: Literal["SPT", "FCFS", "EDD", "MWKR", "SLACK", "DEVIATION"] = "SPT", add_overlap_to_conflict: bool = True):
        schedule_job_collection = LiveJobCollection()
        planned = 0
        while planned < self.total_ops:
            machine_candidates = self.get_machine_candidates()
            all_candidates = [cand for ops in machine_candidates.values() for cand in ops]

            if not all_candidates:
                # Nichts planbar -> hier ggf. Zeit vorspulen oder sauber abbrechen
                break

            earliest_end_t = min(op.end for op in all_candidates)

            for machine, ops in machine_candidates.items():
                ending_at_T = [op for op in ops if op.end == earliest_end_t]
                if not ending_at_T:
                    continue

                conflict_ops = ending_at_T.copy()

                if add_overlap_to_conflict:
                    for o in ops:
                        if o in ending_at_T:
                            continue
                        # überlappt mit mind. einem aus ending_at_T?
                        if any(self.intervals_overlap(o, ending_op) for ending_op in ending_at_T):
                            conflict_ops.append(o)

                # Auswahl nach Prioritätsregel
                selected_op = self.select_by_priority(conflict_ops, priority_rule)

                # print(f"Selected: {selected_op.start = }, {selected_op.end = }")

                if selected_op is not None:

                    job = selected_op.job
                    job.current_operation = job.get_next_operation(selected_op.position_number)
                    job.current_operation_earliest_start = selected_op.end

                    planned += 1
                    self.machine_ready_time[machine] = selected_op.end

                    schedule_job_collection.add_operation_instance(selected_op)

        return schedule_job_collection





    def solve_rolling(
            self, priority_rule: Literal["SPT", "FCFS", "EDD", "MWKR"] = "SPT",
            previous_schedule_jobs_collection: Optional[LiveJobCollection] = None,
            active_jobs_collection: Optional[LiveJobCollection] = None,
            add_overlap_to_conflict: bool = True):

        schedule_job_collection = LiveJobCollection()
        planned = 0
        while planned < self.total_ops:
            machine_candidates = self.get_machine_candidates()
            all_candidates = [cand for ops in machine_candidates.values() for cand in ops]

            if not all_candidates:
                # Nichts planbar -> hier ggf. Zeit vorspulen oder sauber abbrechen
                break

            earliest_end_t = min(op.end for op in all_candidates)

            for machine, ops in machine_candidates.items():
                ending_at_T = [op for op in ops if op.end == earliest_end_t]
                if not ending_at_T:
                    continue

                conflict_ops = ending_at_T.copy()

                if add_overlap_to_conflict:
                    for o in ops:
                        if o in ending_at_T:
                            continue
                        # überlappt mit mind. einem aus ending_at_T?
                        if any(self.intervals_overlap(o, ending_op) for ending_op in ending_at_T):
                            conflict_ops.append(o)

                # Auswahl nach Prioritätsregel
                selected_op = self.select_by_priority(conflict_ops, priority_rule)

                # print(f"Selected: {selected_op.start = }, {selected_op.end = }")

                if selected_op is not None:

                    job = selected_op.job
                    job.current_operation = job.get_next_operation(selected_op.position_number)
                    job.current_operation_earliest_start = selected_op.end

                    planned += 1
                    self.machine_ready_time[machine] = selected_op.end

                    schedule_job_collection.add_operation_instance(selected_op)

        return schedule_job_collection

