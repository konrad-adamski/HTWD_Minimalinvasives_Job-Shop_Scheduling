import pandas as pd

from omega.db_models import RoutingSource, Routing
from omega.db_setup import SessionLocal, reset_tables

if __name__ == "__main__":
    from configs.path_manager import get_path

    reset_tables()

    # RoutingSource
    routing_source = RoutingSource(name="Fisher and Thompson 10x10")


    # Routings
    basic_data_path = get_path("data", "basic")
    df_routings = pd.read_csv(basic_data_path / "ft10_routings.csv")
    routings = Routing.from_multiple_routings_dataframe(df_routings, source=routing_source)

    with SessionLocal() as session:
        session.add_all(routings)
        session.commit()



