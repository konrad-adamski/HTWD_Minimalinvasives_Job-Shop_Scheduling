import numpy as np
import pandas as pd
import warnings

from collections import defaultdict
from decimal import Decimal
from sqlalchemy.exc import IntegrityError, SAWarning
from typing import List, Optional, Dict

from src.Logger import Logger
from src.domain.orm_models import Routing, Experiment, Job, RoutingSource, RoutingOperation, Machine, MachineInstance
from src.domain.orm_setup import SessionLocal

logger = Logger()

class DataSourceInitializer:
    def __new__(cls, *args, **kwargs):
        raise TypeError("RoutingInitializer is a static utility class and cannot be instantiated.")

    @staticmethod
    def insert_from_dictionary(routing_dict: Dict[str, List[Dict[str, int]]], source_name: str):
        """
        Inserts a RoutingSource with Routings, Machines and Operations from a structured dictionary.

        :param routing_dict: e.g. {"0": [{"machine": 4, "duration": 88}, ...], ...}
        :param source_name: Name of the new RoutingSource.
        """
        with SessionLocal() as session:
            try:
                # 1. Neue Quelle anlegen
                source = RoutingSource(name=source_name)
                session.add(source)
                session.flush()

                # 2) Alle benötigten Maschinen-Namen erzeugen
                machine_names = {
                    f"M{op['machine']:02d}"
                    for ops in routing_dict.values()
                    for op in ops
                }

                # 3) Maschinen sofort anlegen (für IDs)
                machines_by_name: dict[str, Machine] = {}
                for name in machine_names:
                    m = Machine(name=name, source=source)
                    session.add(m)
                    machines_by_name[name] = m
                session.flush()

                # 4) Routings + Operationen anlegen
                for routing_id, ops in routing_dict.items():
                    routing_id_str = f"{source.id:02d}-{int(routing_id):02d}"
                    new_routing = Routing(id=routing_id_str, routing_source=source, operations=[])

                    for step_nr, op in enumerate(ops):
                        machine_idx = op["machine"]
                        duration = op["duration"]

                        machine_name = f"M{machine_idx:02d}"
                        machine = machines_by_name[machine_name]

                        new_routing.operations.append(
                            RoutingOperation(
                                routing_id=routing_id_str,
                                position_number=step_nr,
                                machine=machine,
                                duration=duration
                            )
                        )
                    session.add(new_routing)
                session.commit()
                logger.info(f"Data source '{source_name}' Insert successful")
            except Exception as e:
                session.rollback()
                logger.error(f"Data source '{source_name}' Insert failed: {e}")


class JobsInitializer:
    def __new__(cls, *args, **kwargs):
        raise TypeError("JobsInitializer is a static utility class and cannot be instantiated.")

    @staticmethod
    def _get_bottleneck_machine_from_routings(
            routings: List[Routing],
            verbose: bool = False) -> str:
        """
           Identifies the bottleneck machine (with the highest total load) from a list of Routing objects.

           :param routings: List of Routing objects, each containing a list of RoutingOperations
           :param verbose: If True, prints the machine loads
           :return: Name of the machine with the highest total load
           """
        usage = defaultdict(int)

        for routing in routings:
            for op in routing.operations:
                usage[op.machine_name] += op.duration

        if verbose:
            print("Machine workload (total processing time):")
            for machine_name, total in sorted(usage.items(), key=lambda x: str(x[0])):
                print(f"  {machine_name}: {total}")

        if not usage:
            raise ValueError("No machine workload found – list may have been empty or invalid!")

        bottleneck_machine_name = max(usage, key=usage.get)
        if verbose:
            print(f"Bottleneck machine: {bottleneck_machine_name}")

        return bottleneck_machine_name

    @classmethod
    def _get_vec_t_b_mmax_from_routings(cls, routings: List[Routing], verbose: bool = False) -> List[int]:
        """
        Return the processing time of each routing on the bottleneck machine.

        The list is ordered according to the order of the `routings` list.

        :param routings: List of Routing objects, each with a .operations list
        :param verbose: If True, prints total usage per machine
        :return: List of processing times on the bottleneck machine
        """
        # 1) determine the bottleneck machine
        bottleneck_machine = cls._get_bottleneck_machine_from_routings(routings, verbose=verbose)

        # 2) for each routing, sum durations on that machine
        vec = []
        for routing in routings:
            t_b = sum(op.duration for op in routing.operations if op.machine_name == bottleneck_machine)
            vec.append(t_b)
        return vec

    @classmethod
    def _calculate_mean_interarrival_time(
            cls, routings: List[Routing], u_b_mmax: float = 0.9, p: Optional[List[float]] = None,
            verbose: bool = False) -> float:
        """
        Calculates the average interarrival time t_a required to achieve the desired utilization of the bottleneck machine.

        :param routings: List of Routing objects
        :param u_b_mmax: Target utilization of the bottleneck machine (e.g., 0.9)
        :param verbose: If True, prints detailed output
        :return: t_a, rounded to 2 decimal places
        """
        n = len(routings)
        if n == 0:
            raise ValueError("Routingliste darf nicht leer sein.")

        # Processing times on bottleneck machine
        vec_t_b_mmax = cls._get_vec_t_b_mmax_from_routings(routings, verbose=verbose)

        if verbose:
            print(f"Processing times on the bottleneck machine: {vec_t_b_mmax}")

        # 3)  Expected processing time / target utilization
        if p is not None and len(p) == len(vec_t_b_mmax):
            t_a = sum(p[i] * vec_t_b_mmax[i] for i in range(n)) / float(u_b_mmax)
        else:
            # Equal weight routings - probability p = [1.0 / n] * n
            t_a = np.mean(vec_t_b_mmax) / float(u_b_mmax)
        return round(t_a, 4)


    @staticmethod
    def _gen_arrivals(
            mean_interarrival_time: float, size: int, start_time: int = 0,
            last_arrival_time: int = 1440, random_seed: Optional[int] = 120) -> List[int]:

        if random_seed is not None:
            np.random.seed(random_seed)

        # 1) Generate exponentially distributed interarrival times
        interarrival_times = np.random.exponential(scale=mean_interarrival_time, size=size)
        interarrival_times[0] = start_time

        # 2) Calculate cumulative arrival times
        arrivals = np.cumsum(interarrival_times)
        arrivals = np.floor(arrivals).astype(int)

        # 3) Remove values greater than last_arrival_time
        arrivals = arrivals[arrivals <= last_arrival_time]

        return np.floor(arrivals).astype(int).tolist()


    @classmethod
    def _create_jobs(
            cls, routings: List[Routing], max_bottleneck_utilization: float = 0.90, total_shift_number: int = 500,
            arrival_seed: Optional[int] = 120, job_routing_seed: Optional[int] = 100, verbose: bool = False):

        jobs: List[Job] = []

        last_shift_end = 1440 * (total_shift_number + 1)

        # Compute mean interarrival time based on max bottleneck utilization
        mean_arrival_time = cls._calculate_mean_interarrival_time(
            routings=routings,
            u_b_mmax=max_bottleneck_utilization,
            verbose=verbose
        )

        approx_size = np.ceil(last_shift_end / mean_arrival_time).astype(int) + 2 * len(routings)  # with buffer

        arrivals = cls._gen_arrivals(
            mean_interarrival_time=mean_arrival_time,
            size=approx_size,
            last_arrival_time=last_shift_end,
            random_seed=arrival_seed
        )

        a_idx = 0

        max_bottleneck_utilization_db = Decimal(f"{max_bottleneck_utilization:.4f}")
        prefix = f"{max_bottleneck_utilization_db * 10000:05.0f}"
        while a_idx < len(arrivals):
            temp_routings = routings.copy()

            if job_routing_seed is not None:
                np.random.seed(job_routing_seed + a_idx)
                np.random.shuffle(temp_routings)

            for i, routing in enumerate(temp_routings):
                if a_idx >= len(arrivals):
                    break
                job = Job(id=f"{routing.source_id:02d}-{prefix}-{a_idx:04d}",
                          routing=routing,
                          arrival=arrivals[a_idx],
                          due_date=None,
                          max_bottleneck_utilization=max_bottleneck_utilization_db
                          )
                jobs.append(job)
                a_idx += 1
        return jobs

    @classmethod
    def create_simple_jobs(cls, routings: List[Routing], job_routing_seed: Optional[int] = 100, shuffle: bool = True):
        jobs: List[Job] = []
        temp_routings = routings.copy()

        if shuffle:
            if job_routing_seed is not None:
                np.random.seed(job_routing_seed)
            np.random.shuffle(temp_routings)

        for i, routing in enumerate(temp_routings):
            job = Job(id=f"{routing.source_id:02d}-{i:04d}",
                      routing=routing,
                      arrival=None,
                      due_date=None,
                      max_bottleneck_utilization=None
                      )
            jobs.append(job)
        return jobs

    @classmethod
    def insert_jobs(
            cls, routings: List[Routing], max_bottleneck_utilization: float = 0.90, total_shift_number: int = 500,
            arrival_seed: Optional[int] = 120, job_routing_seed: Optional[int] = 100, verbose: bool = False):

        jobs = cls._create_jobs(
            routings=routings,
            max_bottleneck_utilization=max_bottleneck_utilization,
            total_shift_number=total_shift_number,
            arrival_seed=arrival_seed,
            job_routing_seed=job_routing_seed,
            verbose=verbose
        )
        job_number = len(jobs)

        with SessionLocal() as session:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=SAWarning)
                try:
                    session.add_all(jobs)
                    session.commit()
                    logger.info(f"Jobs Insert with {max_bottleneck_utilization = }, {job_number = } successful")
                except SAWarning as w:
                    session.rollback()
                    logger.warning(f"Jobs Insert with {max_bottleneck_utilization = } prevented: {w}")
                except IntegrityError as e:
                    session.rollback()
                    logger.error(f"Jobs Insert with {max_bottleneck_utilization = } failed {e}")


class MachineInstanceInitializer:
    def __new__(cls, *args, **kwargs):
        raise TypeError("MachineInitializer is a static utility class and cannot be instantiated.")

    @staticmethod
    def insert_from_dataframe(
            df: pd.DataFrame, source_name: str, max_bottleneck_utilization: Decimal, machine_column: str = "Machine",
            average_transition_time_column: str = "Ø Transition Time"):
        """
        Inserts Machine entries into the database from a pandas DataFrame.

        For each row in the DataFrame, a new Machine record is created with the given machine name, transition time,
        max bottleneck utilization, and linked to the specified RoutingSource.

        :param df: Pandas DataFrame containing machine data. Must contain at least the
                   columns specified by `machine_column` and `average_transition_time_column`.
        :param source_name: Name of the RoutingSource to associate with the machines.
        :param max_bottleneck_utilization: Maximum bottleneck utilization value to assign to each machine.
        :param machine_column: Name of the column in `df` that contains machine names.
        :param average_transition_time_column: Name of the column in `df` that contains average transition times.
        :raises ValueError: If the RoutingSource does not exist or such machines (with given max_bottleneck_utilization)
                            already exist.
        """

        with SessionLocal() as session:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=SAWarning)
                try:
                    #routing_source = (
                    #    session.query(RoutingSource)
                    #    .filter(RoutingSource.name == source_name)
                    #    .one_or_none()
                    #)

                    for _, row in df.iterrows():
                        machine_name = row[machine_column]
                        machine = (
                            session.query(Machine)
                            .join(Machine.source)
                            .filter(RoutingSource.name == source_name, Machine.name == machine_name)
                            .one_or_none()
                        )
                        machine_instance = MachineInstance(
                            machine = machine,
                            transition_time=int(row[average_transition_time_column]),
                            max_bottleneck_utilization= max_bottleneck_utilization,
                        )
                        session.add(machine_instance)
                    session.commit()
                    logger.info(f"Machines Insert {source_name = } {max_bottleneck_utilization = } successful")
                except SAWarning as w:
                    session.rollback()
                    logger.warning(f"Machine Insert {source_name = } {max_bottleneck_utilization = } prevented: {w}")
                except IntegrityError as e:
                    session.rollback()
                    logger.error(f"Machine Insert {source_name = } {max_bottleneck_utilization = } failed: {e}")



class ExperimentInitializer:
    def __new__(cls, *args, **kwargs):
        raise TypeError("ExperimentInitializer is a static utility class and cannot be instantiated.")

    @staticmethod
    def insert_experiment(
            source_name: str, absolute_lateness_ratio: float, inner_tardiness_ratio: float,
            max_bottleneck_utilization: Decimal, sim_sigma: float,
            experiment_type: Optional[str] = None) -> Optional[int]:
        """
        Inserts a single Experiment entry into the database.

        :param source_name: Name of the RoutingSource to associate with the experiment.
        :param absolute_lateness_ratio: Ratio for absolute lateness weight.
        :param inner_tardiness_ratio: Ratio for inner tardiness weight.
        :param max_bottleneck_utilization: Maximum bottleneck utilization value.
        :param sim_sigma: Sigma value for simulation variability.
        :param experiment_type: Type of experiment.
        :return: experiment_id if the entry was inserted successfully
        """

        with SessionLocal() as session:
            try:
                routing_source = (
                    session.query(RoutingSource)
                    .filter(RoutingSource.name == source_name)
                    .one_or_none()
                )
                experiment = Experiment(
                    routing_source=routing_source,
                    absolute_lateness_ratio=absolute_lateness_ratio,
                    inner_tardiness_ratio=inner_tardiness_ratio,
                    max_bottleneck_utilization=max_bottleneck_utilization,
                    sim_sigma=sim_sigma,
                    type=experiment_type
                )
                session.add(experiment)
                session.flush()
                experiment_id = experiment.id
                session.commit()
                logger.info(f"Experiment insert successful for {source_name = }, {max_bottleneck_utilization = }.")
                return experiment_id
            except Exception as e:
                session.rollback()
                logger.error(f"Experiment insert failed for {source_name = }, {max_bottleneck_utilization = }: {e}")
                return None

