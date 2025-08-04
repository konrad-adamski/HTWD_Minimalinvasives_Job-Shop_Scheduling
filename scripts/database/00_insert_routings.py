from typing import List

import pandas as pd

from src.classes.Query import RoutingQuery
from src.classes.orm_models import RoutingSource, Routing
from src.classes.orm_setup import reset_tables

if __name__ == "__main__":
    from configs.path_manager import get_path

    reset_tables()

    # RoutingSource
    source_name = "Fisher and Thompson 10x10"
    routing_source = RoutingSource(name="Fisher and Thompson 10x10")


    # Routings
    basic_data_path = get_path("data", "basic")
    df_routings = pd.read_csv(basic_data_path / "ft10_routings.csv")

    RoutingQuery.insert_from_dataframe(df_routings = df_routings, source=routing_source)

    routings: List[Routing] = RoutingQuery.get_by_source_name(source_name=source_name)

    for routing in routings:
        print(routing)



