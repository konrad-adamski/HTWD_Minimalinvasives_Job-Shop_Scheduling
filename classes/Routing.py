from collections import UserDict
from typing import Optional
import pandas as pd

from classes.Operation import RoutingOperation


class RoutingOperationCollection(UserDict):
    """
    Stores routing operations by routing ID, sorted by sequence number.
    """
    def add_operation(self, routing_id: str, sequence_number: int, machine: str, duration: int):
        """
        Adds a RoutingOperation to the collection.
        """
        if routing_id not in self:
            self[routing_id] = []
        self[routing_id].append(RoutingOperation(sequence_number, machine, duration))

    def sort_operations(self):
        """
        Sorts operations within each routing by sequence number.
        """
        for routing_id in self:
            self[routing_id].sort(key=lambda op: op.sequence_number)

    def get_operation(self, routing_id: str, sequence_number: int) -> Optional[RoutingOperation]:
        """
        Returns a specific RoutingOperation if it exists.
        """
        return next((op for op in self[routing_id] if op.sequence_number == sequence_number), None)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, routing_column: str = "Routing_ID", operation_column: str = "Operation",
                       machine_column: str = "Machine", duration_column: str = "Processing Time"):
        """
        Creates a collection from DataFrame; drops duplicates.
        """
        obj = cls()
        df_clean = df.drop_duplicates(subset=[routing_column, operation_column], keep="first")
        for _, row in df_clean.iterrows():
            obj.add_operation(
                routing_id = str(row[routing_column]),
                sequence_number = int(row[operation_column]),
                machine = str(row[machine_column]),
                duration=int(row[duration_column])
            )

        obj.sort_operations()
        return obj

    def to_dataframe(
            self, routing_column: str = "Routing_ID", operation_column: str = "Operation",
            machine_column: str = "Machine", duration_column: str = "Processing Time") -> pd.DataFrame:
        """
        Converts the collection to a DataFrame.

        :param routing_column: Column name for routing ID
        :param operation_column: Column name for sequence number
        :param machine_column: Column name for machine
        :param duration_column: Column name for duration
        :return: DataFrame representation of routing operations
        """
        rows = []
        for routing_id, ops in self.items():
            for op in ops:
                rows.append({
                    routing_column: routing_id,
                    operation_column: op.sequence_number,
                    machine_column: op.machine,
                    duration_column: op.duration
                })
        return pd.DataFrame(rows)

if __name__ == "__main__":

    # Examples (with duplikates)
    df = pd.DataFrame([
        {"Routing_ID": "R1", "Operation": 0, "Machine": "M1", "Processing Time": 5},
        {"Routing_ID": "R1", "Operation": 1, "Machine": "M2", "Processing Time": 3},
        {"Routing_ID": "R1", "Operation": 1, "Machine": "M2_DUP", "Processing Time": 999},
        {"Routing_ID": "R2", "Operation": 0, "Machine": "M3", "Processing Time": 4},
        {"Routing_ID": "R2", "Operation": 1, "Machine": "M1", "Processing Time": 6},
    ])

    # Routing collection from DataFrame
    routing_collection = RoutingOperationCollection.from_dataframe(df)

    # Output
    print("\n--- RoutingOperationCollection ---")
    for routing_id, operations in routing_collection.items():
        print(f"Routing: {routing_id}")
        for op in operations:
            print(f"  Operation {op.sequence_number}: Machine {op.machine}, Duration {op.duration}")
