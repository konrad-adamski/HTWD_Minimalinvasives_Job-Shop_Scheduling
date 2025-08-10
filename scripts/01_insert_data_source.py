import json

from project_config import get_data_path
from src.domain.Initializer import DataSourceInitializer
from src.domain.orm_setup import reset_tables

if __name__ == "__main__":
    reset_tables()

    # Load file
    file_path = get_data_path("basic", "jobshop_instances.json")

    with open(file_path, "r", encoding="utf-8") as f:
        jobshop_instances = json.load(f)

    # Extract "Fisher and Thompson 10x10" job (routing) shop scheduling problem
    data_source: dict = jobshop_instances["instance ft10"]

    print("\n" + "-"*30, "Original data source", "-"*30)
    for key,value in data_source.items():
        print(f"{key}: {value}")

    print("-"*82)

    # RoutingSource with Routings, RoutingOperations and Machines
    source_name = "Fisher and Thompson 10x10"
    DataSourceInitializer.insert_from_dictionary(data_source, source_name = source_name)
