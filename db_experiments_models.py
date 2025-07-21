
from database import analysis_db as db

from peewee import Model, CharField, IntegerField, CompositeKey, FloatField
from playhouse.shortcuts import chunked

import pandas as pd

class ScheduleEntry(Model):
    job = CharField()
    routing_id = IntegerField()
    arrival = IntegerField()
    ready_time = IntegerField()
    deadline = IntegerField()
    operation = IntegerField()
    machine = CharField()
    start = IntegerField()
    processing_time = IntegerField()
    end = IntegerField()
    date = IntegerField()
    version = CharField()

    class Meta:
        database = db
        primary_key = CompositeKey('job', 'operation', 'date', 'version')  # eindeutige Kombination


    @classmethod
    def add_schedule_from_dataframe(cls, df: pd.DataFrame, date: int, version: str):
        """
        Speichert jede Zeile eines DataFrames als separaten ScheduleEntry.
        Überschreibt bestehende Einträge mit gleichem Schlüssel (job, operation, date, version).
        """
        entries = []
        for _, row in df.iterrows():
            entries.append({
                'job': row['Job'],
                'routing_id': row['Routing_ID'],
                'arrival': row['Arrival'],
                'ready_time': row['Ready Time'],
                'deadline': row['Deadline'],
                'operation': row['Operation'],
                'machine': row['Machine'],
                'start': row['Start'],
                'processing_time': row['Processing Time'],
                'end': row['End'],
                'date': date,
                'version': version
            })

        with db.atomic():
            for batch in chunked(entries, 100):
                cls.insert_many(batch).on_conflict_replace().execute()

        print(f"✅ {len(entries)} Schedule-Einträge gespeichert (mit Überschreiben) – Version={version}, Date={date}")

    @classmethod
    def get_schedule_as_dataframe(cls, date: int, version: str) -> pd.DataFrame:
        """
        Lädt alle Schedule-Einträge für gegebenes Datum und Version.
        """
        query = (cls
                 .select()
                 .where((cls.date == date) & (cls.version == version)))

        if not query.exists():
            print(f"⚠️ Keine Einträge gefunden für Date={date}, Version='{version}'.")
            return pd.DataFrame()

        df = pd.DataFrame(list(query.dicts()))
        return df.drop(columns=['date', 'version'])

    @classmethod
    def clone_schedule_entries(cls, referenced_version: str, new_version: str):
        """
        Klont alle Einträge einer gegebenen Referenz-Version in eine neue Version.
        Vor dem Kopieren werden ggf. vorhandene Einträge mit der neuen Version gelöscht.
        """
        # 1. Alte Version löschen
        delete_query = cls.delete().where(cls.version == new_version)
        deleted = delete_query.execute()
        if deleted > 0:
            print(f"🗑️ {deleted} alte Einträge mit Version='{new_version}' gelöscht.")

        # 2. Neue Version erzeugen aus Referenz
        entries_to_clone = cls.select().where(cls.version == referenced_version)
        if not entries_to_clone.exists():
            print(f"⚠️ Keine Einträge mit Version='{referenced_version}' gefunden.")
            return

        count = 0
        with db.atomic():
            for batch in chunked(entries_to_clone, 100):
                data = []
                for row in batch:
                    row_dict = row.__data__.copy()
                    row_dict["version"] = new_version
                    data.append(row_dict)
                cls.insert_many(data).execute()
                count += len(data)

        print(f"✅ {count} Einträge von Version='{referenced_version}' nach Version='{new_version}' kopiert.")



class Execution(Model):
    job = CharField()
    routing_id = IntegerField(null=True)
    operation = IntegerField()
    machine = CharField()
    arrival = IntegerField()
    start = FloatField()
    processing_time = FloatField()
    end = FloatField()
    date = IntegerField()
    version = CharField()

    class Meta:
        database = db
        primary_key = CompositeKey('job', 'operation', 'date', 'version')




    @classmethod
    def add_executions_from_dataframe(cls, df: pd.DataFrame, date: int, version: str):
        """
        Speichert jede Zeile eines DataFrames als separaten Execution-Eintrag.
        Überschreibt bestehende Einträge mit gleichem Schlüssel (job, operation, date, version).
        """
        entries = []
        for _, row in df.iterrows():
            entries.append({
                'job': row['Job'],
                'routing_id': row['Routing_ID'] if not pd.isna(row['Routing_ID']) else None,
                'operation': row['Operation'],
                'machine': row['Machine'],
                'arrival': int(row['Arrival']),
                'start': float(row['Start']),
                'processing_time': float(row['Processing Time']),
                'end': float(row['End']),
                'date': date,
                'version': version
            })

        with db.atomic():
            for batch in chunked(entries, 100):
                cls.insert_many(batch).on_conflict_replace().execute()

        print(f"✅ {len(entries)} Execution-Einträge gespeichert (mit Überschreiben) – Version={version}, Date={date}")

    @classmethod
    def get_executions_as_dataframe(cls, date: int, version: str) -> pd.DataFrame:
        """
        Gibt die Execution-Einträge als DataFrame zurück, gefiltert nach date und version.
        """
        query = cls.select().where((cls.date == date) & (cls.version == version))
        if not query.exists():
            print(f"⚠️ Keine Execution-Einträge gefunden für Date={date}, Version='{version}'.")
            return pd.DataFrame()

        df = pd.DataFrame(list(query.dicts()))
        return df.drop(columns=['date', 'version'])

class Active(Model):
    job = CharField()
    operation = IntegerField()
    machine = CharField()
    arrival = IntegerField()
    start = FloatField()
    planned_duration = IntegerField()
    processing_time = FloatField()
    expected_end = FloatField()
    end = FloatField()
    date = IntegerField()
    version = CharField()

    class Meta:
        database = db
        primary_key = CompositeKey('job', 'operation', 'date', 'version')


@classmethod
def add_active_from_dataframe(cls, df: pd.DataFrame, date: int, version: str):
    """
    Speichert jede Zeile eines DataFrames als separaten Active-Eintrag.
    Überschreibt bestehende Einträge mit gleichem Schlüssel (job, operation, date, version).
    """
    entries = []
    for _, row in df.iterrows():
        entries.append({
            'job': row['Job'],
            'operation': row['Operation'],
            'machine': row['Machine'],
            'arrival': int(row['Arrival']),
            'start': float(row['Start']),
            'planned_duration': int(row['Planned Duration']),
            'processing_time': float(row['Processing Time']),
            'expected_end': float(row['Expected End']),
            'end': float(row['End']),
            'date': date,
            'version': version
        })

    with db.atomic():
        for batch in chunked(entries, 100):
            cls.insert_many(batch).on_conflict_replace().execute()

    print(f"✅ {len(entries)} Active-Einträge gespeichert (mit Überschreiben) – Version={version}, Date={date}")

@classmethod
def get_active_as_dataframe(cls, date: int, version: str) -> pd.DataFrame:
    """
    Gibt die Active-Einträge als DataFrame zurück, gefiltert nach date und version.
    """
    query = cls.select().where((cls.date == date) & (cls.version == version))
    if not query.exists():
        print(f"⚠️ Keine Active-Einträge gefunden für Date={date}, Version='{version}'.")
        return pd.DataFrame()

    df = pd.DataFrame(list(query.dicts()))
    return df.drop(columns=['date', 'version'])
