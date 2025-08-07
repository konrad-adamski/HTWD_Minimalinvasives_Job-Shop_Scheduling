from collections import UserDict
from typing import Set, Optional, Iterable

import simpy

class MachineSource(simpy.Resource):
    def __init__(self, env, name):
        super().__init__(env, capacity=1)
        self.name = name

class SimulationMachine:
    """
    Wrapper for a single machine instance that can be bound to a SimPy environment later.
    """
    def __init__(self, name: str, env: Optional[simpy.Environment] = None):
        self.name = name
        self.env = env
        self.source: Optional[MachineSource] = None

        if env is not None:
            self.source = MachineSource(env, name)

    def reload(self, new_env: simpy.Environment):
        """
        Reinitializes the machine with a new environment.
        """
        self.env = new_env
        self.source = MachineSource(new_env, self.name)

    def request(self):
        if self.source is None:
            raise RuntimeError(f"Machine {self.name} has not been initialized with an environment.")
        return self.source.request()

    def __str__(self):
        return f"<Machine {self.name}>"


class SimulationMachineCollection(UserDict):
    """
    A dictionary-like collection managing Machine wrapper instances by name.
    """
    def add_machine(self, machine_name: str):
        """
        Adds a Machine by name if not already present.
        The Machine will initially be uninitialized (without env).
        """
        if machine_name not in self.data:
            self.data[machine_name] = SimulationMachine(machine_name)

    def add_machines(self, machine_names: Iterable[str]):
        """
         Adds multiple Machines by name if not already present.
         Each Machine will initially be uninitialized (without env).

         :param machine_names: A set or list of machine names to add.
         """
        for name in machine_names:
            self.add_machine(name)

    def set_env(self, new_env: simpy.Environment):
        """
        Sets the environment for all machines by reinitializing their internal resources.

        :param new_env: The SimPy environment to apply to all machines.
        """
        for machine in self.data.values():
            machine.reload(new_env)

    def add_machines_with_env(self, env: simpy.Environment, machine_names: Set[str]):
        """
        Adds machines (if not already present) and sets their environment.

        :param env: SimPy environment to bind the machines to
        :param machine_names: Set or list of machine names to ensure exist
        """
        self.add_machines(machine_names)
        self.set_env(env)

    def get_machine(self, name: str) -> SimulationMachine:
        """
        Returns the Machine instance for the given name (not Resource).

        :param name: Name of the machine.
        :return: Machine wrapper instance.
        :raises KeyError: If the machine is not found in the collection.
        """
        return self.data[name]

    def get_source(self, name: str) -> MachineSource:
        """
        Returns the simpy.Resource (MachineSource) of the given machine.

        :param name: Name of the machine.
        :return: SimPy resource.
        :raises RuntimeError: If the machine is not initialized with an environment.
        """
        machine = self.get_machine(name)
        if machine.source is None:
            raise RuntimeError(f"Machine '{name}' is not initialized with an environment.")
        return machine.source
