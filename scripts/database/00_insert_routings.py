from typing import List

import pandas as pd

from src.classes.Collections import RoutingsCollection
from src.classes.orm_models import RoutingSource, Routing
from src.classes.orm_setup import SessionLocal, reset_tables

if __name__ == "__main__":
    from configs.path_manager import get_path

    reset_tables()

    # RoutingSource
    source_name = "Fisher and Thompson 10x10"
    routing_source = RoutingSource(name="Fisher and Thompson 10x10")


    # Routings
    basic_data_path = get_path("data", "basic")
    df_routings = pd.read_csv(basic_data_path / "ft10_routings.csv")

    routing_collection = RoutingsCollection.from_dataframe(
        df_routings = df_routings,
        source=routing_source
    )

    routings: List[Routing] = routing_collection.get_routings()

    with SessionLocal() as session:
        session.add_all(routings)
        session.commit()
        routing_collection = RoutingsCollection()

    routing_collection = RoutingsCollection.from_db_by_source_name(source_name= source_name)
    for routing in routing_collection.values():
        print(routing)



