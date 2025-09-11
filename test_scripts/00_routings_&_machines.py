import time

from src.domain.Query import RoutingQuery, MachineQuery

if __name__ == '__main__':
    routings = RoutingQuery.get_by_source_name("Fisher and Thompson 10x10")

    for routing in routings[:3]:
        print(routing)
        for operation in routing.operations[:2]:
            machine_name = operation.machine.name
            print(f"\t{operation} {machine_name = }")

    print("--"*60)
    start = time.time()
    machines = MachineQuery.get_by_source_name("Fisher and Thompson 10x10")
    end = time.time()
    print(f"Duration: {end - start}")
    for machine in machines:
        print(machine)
