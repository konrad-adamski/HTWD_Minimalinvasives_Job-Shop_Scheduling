import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, joinedload

from omega.db_models import mapper_registry, RoutingSource, Routing

if __name__ == "__main__":

    # Example with multiple Routings
    data = {
        "Routing_ID": ["R1", "R1", "R2"],
        "Operation": [10, 20, 10],
        "Machine": ["M1", "M2", "M3"],
        "Processing Time": [5, 10, 7]
    }
    dframe_routings = pd.DataFrame(data)


    # SQLite-Datenbank (lokal in Datei)
    engine = create_engine("sqlite:///test.db")  # oder sqlite:///:memory: (im RAM)

    # ❌ Alles löschen
    mapper_registry.metadata.drop_all(engine)
    # ✅ Neu erzeugen
    mapper_registry.metadata.create_all(engine)

    session = Session(engine)

    # Arbeiten mit session
    routing_source = RoutingSource(name="Mein Test")
    session.add(routing_source)
    session.flush()  # wichtig, damit ID verfügbar



    routings = Routing.from_multiple_routings_dataframe(
        dframe_routings,
        source=routing_source
    )
    session.add_all(routings)
    session.commit()

    session.close()  # <- nicht vergessen!


    routings = session.query(Routing).options(
        joinedload(getattr(Routing, "routing_source")),
        joinedload(getattr(Routing, "operations")),
        joinedload(getattr(Routing, "jobs"))
    ).all()

    for routing in routings:
        print(f"Routing-ID: {routing.id} from {routing.routing_source}")
        print(f"Gesamtdauer: {routing.sum_duration} min")

        for op in routing.operations:
            for op in routing.operations:
                print(f"  • Step {op.position_number}: {op.machine}, {op.duration} min")

