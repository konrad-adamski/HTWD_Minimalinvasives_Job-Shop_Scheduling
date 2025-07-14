import pandas as pd
# Datenzugriff
from configs.path_manager import get_path
# Database
from models import *

def create_routings_and_operations(df_instance: pd.DataFrame):
    grouped = df_instance.groupby("Routing_ID")

    for routing_id, group in grouped:
        # 1. Routing ohne Beschreibung (Konflikte ignorieren)
        try:
            Routing.insert(id=str(routing_id)).on_conflict_ignore().execute()
            print(f"Routing '{routing_id}' wurde angelegt oder war bereits vorhanden.")
        except Exception as e:
            print(f"Fehler bei Routing '{routing_id}': {e}")

        # 2. Alle zugehörigen Operationen (Konflikte ignorieren)
        for _, row in group.iterrows():
            try:
                RoutingOperation.insert(
                    routing=str(row["Routing_ID"]),
                    operation=int(row["Operation"]),
                    machine=str(row["Machine"]),
                    duration=int(row["Processing Time"])
                ).on_conflict_ignore().execute()
            except Exception as e:
                print(f"Fehler bei Operation {row['Operation']} von Routing {routing_id}: {e}")

if __name__ == "__main__":

    # Datei laden
    basic_data_path = get_path("data", "basic")
    file_path = basic_data_path / "instance.csv"

    df_instances = pd.read_csv(file_path)
    print(df_instances.head(10))

    reset_tables()
    create_routings_and_operations(df_instances)
