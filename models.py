import pandas as pd
from peewee import Model, TextField, CharField, ForeignKeyField, IntegerField, AutoField, BooleanField, CompositeKey, \
    FloatField
from playhouse.sqlite_ext import JSONField

from database import db


class Routing(Model):
    routing = CharField(verbose_name="Routing ID")         # Teil des Composite Keys
    operation = IntegerField(verbose_name="Operation")      # Teil des Composite Keys
    machine = CharField(verbose_name="Machine")
    duration = IntegerField(verbose_name="Duration")

    class Meta:
        database = db
        primary_key = CompositeKey('routing', 'operation')  # 👈 Composite Primary Key

    @classmethod
    def add_routing(cls, routing_id: str, operation: int, machine: str, duration: int):
        """
        Fügt eine Routing-Zeile hinzu, sofern Kombination aus Routing-ID und Operation noch nicht existiert.
        """
        try:
            cls.create(
                routing=routing_id,
                operation=operation,
                machine=machine,
                duration=duration
            )
            print(f"✅ Routing '{routing_id}', Operation {operation} wurde hinzugefügt.")
        except Exception as e:
            print(f"❌ Fehler beim Hinzufügen von Routing '{routing_id}', Operation {operation}: {e}")

    @classmethod
    def get_dataframe(cls) -> pd.DataFrame:
        """
        Gibt alle Routing-Datensätze als Pandas DataFrame zurück.
        """
        data = [
            {
                "Routing_ID": r.routing,
                "Operation": r.operation,
                "Machine": r.machine,
                "Processing Time": r.duration
            }
            for r in cls.select().order_by(cls.routing, cls.operation)
        ]
        return pd.DataFrame(data)

class Job(Model):
    job_id = CharField(verbose_name="Job ID")
    routing = CharField(verbose_name="Routing ID")
    arrival_time = IntegerField(verbose_name="Arrival Time")
    ready_time = IntegerField(verbose_name="Ready Time")
    due_date = IntegerField(verbose_name="Due Date")
    status = CharField(default="open", verbose_name="Status")
    version = CharField(default="base", verbose_name="Version")

    class Meta:
        database = db
        primary_key = CompositeKey('job_id', 'version')  # Composite Primary Key

    @classmethod
    def add_jobs_from_dataframe(cls, df: pd.DataFrame, routing_column='Routing_ID', due_date_column='Deadline',
                                version: str = "base", status: str = "open"):
        """
        Fügt Jobs mit Composite Primary Key (job_id, version) ein oder aktualisiert sie.
        """
        count_change = 0
        for _, row in df.iterrows():
            try:
                routing_id = str(row[routing_column])
                if not Routing.select().where(Routing.routing == routing_id).exists():
                    print(f"⚠️ Routing-ID '{routing_id}' nicht gefunden – Job {row['Job']} wird übersprungen.")
                    continue

                data = {
                    "job_id": str(row['Job']),
                    "version": version,
                    "routing": routing_id,
                    "arrival_time": int(row['Arrival']),
                    "ready_time": int(row['Ready Time']),
                    "due_date": int(row[due_date_column]),
                    "status": status
                }

                cls.insert(data).on_conflict(
                    conflict_target=['job_id', 'version'],
                    update=data
                ).execute()

                count_change += 1
            except Exception as e:
                print(f"❌ Fehler bei Job {row['Job']}: {e}")
        print(f"✅ {count_change} Jobs (Version '{version}') wurden hinzugefügt oder aktualisiert.")

    @classmethod
    def get_dataframe(cls, version: str = None, arrival_time_min: int = None, arrival_time_max: int = None,
                      ready_time_is: int = None, status: str = None) -> pd.DataFrame:
        """
        Gibt alle Jobs als DataFrame zurück.
        Optional kann nach Version, Status, Ready Time (genau) und Arrival Time (Bereich) gefiltert werden.
        """
        query = cls.select()
        if version is not None:
            query = query.where(cls.version == version)
        if status is not None:
            query = query.where(cls.status == status)
        if ready_time_is is not None:
            query = query.where(cls.ready_time == ready_time_is)
        if arrival_time_min is not None:
            query = query.where(cls.arrival_time >= arrival_time_min)
        if arrival_time_max is not None:
            query = query.where(cls.arrival_time < arrival_time_max)

        data = [
            {
                "Job": job.job_id,
                "Routing_ID": job.routing,
                "Arrival": job.arrival_time,
                "Ready Time": job.ready_time,
                "Deadline": job.due_date,
                "Status": job.status,
                "Version": job.version,
            }
            for job in query
        ]

        return pd.DataFrame(data)

    @classmethod
    def clone_jobs(cls, referenced_version: str, new_version: str):
        """
        Klont alle Jobs aus der angegebenen Referenzversion in eine neue Version.
        Die Job-IDs bleiben gleich, nur die Version wird ersetzt.
        Überschreibt existierende Einträge mit (job_id, new_version).
        """
        jobs = cls.select().where(cls.version == referenced_version)
        count = 0

        for job in jobs:
            try:
                data = {
                    "job_id": job.job_id,
                    "version": new_version,
                    "routing": job.routing,
                    "arrival_time": job.arrival_time,
                    "ready_time": job.ready_time,
                    "due_date": job.due_date,
                    "status": job.status,
                }

                cls.insert(data).on_conflict(
                    conflict_target=["job_id", "version"],
                    update=data
                ).execute()
                count += 1
            except Exception as e:
                print(f"❌ Fehler beim Klonen von Job {job.job_id}: {e}")

        print(f"✅ {count} Jobs von Version '{referenced_version}' nach Version '{new_version}' kopiert.")



class Schedule(Model):
    id = AutoField(verbose_name="ID")
    date = IntegerField(null=True, verbose_name="Date")
    data = JSONField(verbose_name="Data")
    version = CharField(null=True, verbose_name="Version")

    class Meta:
        database = db

    @classmethod
    def add_schedule(cls, data: dict, date: int = None, version: str = None):
        """
        Fügt einen neuen Schedule-Eintrag in die Datenbank ein.

        :param data: Der JSON-ähnliche Inhalt des Schedules
        :param date
        :param version
        """
        try:
            cls.create(
                data=data,
                date=date,
                version=version
            )
            print(f"✅ Schedule hinzugefügt (Version={version}, Date={date})")
        except Exception as e:
            print(f"❌ Fehler beim Hinzufügen des Schedules: {e}")

    @classmethod
    def get_schedule_as_dataframe(cls, date: int, version: str = None) -> pd.DataFrame:
        """
        Gibt das Schedule-JSON als DataFrame zurück, gefiltert nach Datum und optional Version und Beschreibung.
        Falls mehrere Einträge passen, wird der letzte (höchste ID) verwendet.
        """
        query = cls.select().where(cls.date == date)

        if version is not None:
            query = query.where(cls.version == version)

        query = query.order_by(cls.id.desc())
        schedules = list(query)

        if not schedules:
            print(f"⚠️ Kein Schedule gefunden für Date={date}, Version='{version}'.")
            return pd.DataFrame()

        if len(schedules) > 1:
            print(
                f"ℹ️ Achtung: {len(schedules)} Schedules gefunden – letzter Eintrag (ID {schedules[0].id}) wird verwendet.")

        try:
            return pd.DataFrame(schedules[0].data)
        except Exception as e:
            print(f"❌ Fehler beim Umwandeln in DataFrame: {e}")
            return pd.DataFrame()

    @classmethod
    def clone_schedules(cls, referenced_version: str, new_version: str):
        """
        Klont alle Schedule-Einträge der angegebenen Referenz-Version in eine neue Version.
        Datum bleibt erhalten. Überspringt Einträge, wenn (date, version) bereits existieren.
        """
        schedules = cls.select().where(cls.version == referenced_version)

        if not schedules:
            print(f"⚠️ Keine Schedules gefunden für Version='{referenced_version}'.")
            return

        count = 0
        for s in schedules:
            try:
                cls.insert({
                    "date": s.date,
                    "data": s.data,
                    "version": new_version
                }).on_conflict_ignore().execute()  # Duplikate vermeiden
                count += 1
            except Exception as e:
                print(f"❌ Fehler beim Klonen (Date={s.date}): {e}")

        print(f"✅ {count} Schedule-Einträge von Version '{referenced_version}' nach '{new_version}' kopiert.")



# für Simulation
class JobOperation(Model):
    job_id = CharField(verbose_name="Job ID")       # Teil des Composite Foreign Key
    operation = IntegerField(verbose_name="Operation")  # Teil des Composite Primary Key
    machine = CharField(verbose_name="Machine")
    start = FloatField(null=True, verbose_name="Start Time")
    processing_time = FloatField(verbose_name="Processing Time")
    end = FloatField(null=True, verbose_name="End Time")
    status = CharField(default="open", verbose_name="Status")
    version = CharField(verbose_name="Version")  # Teil des Composite Foreign Key

    class Meta:
        database = db
        primary_key = CompositeKey('job_id', 'version', 'operation')  # 👈 Composite PK


    @classmethod
    def add_from_dataframe(cls, df: pd.DataFrame, version: str = "base", status: str = "open"):
        """
        Fügt JobOperation-Einträge aus einem DataFrame ein oder aktualisiert sie.
        Erwartete Spalten: 'Job', 'Operation', 'Machine'. Optional: 'Start', 'End'.
        'version' und 'status' werden einheitlich übergeben.
        """
        count = 0
        for _, row in df.iterrows():
            try:
                data = {
                    "job_id": str(row["Job"]).strip(),
                    "machine": str(row["Machine"]).strip(),
                    "operation": int(row["Operation"]),
                    "processing_time": float(row["Processing Time"]),
                    "version": version,
                    "status": status
                }

                # Optional: Start/End nur setzen, wenn vorhanden und gültig
                if "Start" in row and pd.notna(row["Start"]):
                    data["start"] = float(row["Start"])
                if "End" in row and pd.notna(row["End"]):
                    data["end"] = float(row["End"])

                cls.insert(data).on_conflict(
                    conflict_target=["job_id", "version", "operation"],
                    update=data
                ).execute()

                count += 1
            except Exception as e:
                print(f"❌ Fehler bei JobOperation ({row.get('Job')}, {version}, {row.get('Operation')}): {e}")
        print(
            f"✅ {count} JobOperation-Einträge (Version '{version}', Status '{status}') wurden hinzugefügt oder aktualisiert.")

    @classmethod
    def get_dataframe(cls, version: str, jobs: list[str] = None, status: str = None) -> pd.DataFrame:
        """
        Gibt einen DataFrame aller JobOperationen für eine bestimmte Version zurück.
        Optional kann zusätzlich nach Job-IDs und Status gefiltert werden.
        """
        query = cls.select().where(cls.version == version)

        if jobs is not None and len(jobs) > 0:
            query = query.where(cls.job_id.in_(list(jobs)))

        if status is not None:
            query = query.where(cls.status == status)

        data = [
            {
                "Job": row.job_id,
                "Machine": row.machine,
                "Operation": row.operation,
                "Start": row.start,
                "End": row.end,
                "Processing Time": row.processing_time,
                "Version": row.version,
                "Operation Status": row.status
            }
            for row in query.order_by(cls.job_id, cls.operation)
        ]

        return pd.DataFrame(data)

    @classmethod
    def update_closed_jobs_from_operations(cls, version: str) -> list[str]:
        """
        Setzt den Status in der Job-Tabelle auf 'closed' für alle Jobs, deren Operationen vollständig 'finished' sind.
        Gilt nur für Jobs in der übergebenen Version.
        Rückgabe: Liste der betroffenen Job-IDs.
        """
        # Schritt 1: Alle Operationen der Version holen
        query = cls.select(cls.job_id, cls.status).where(cls.version == version)

        job_status_counts = {}
        for row in query:
            job = row.job_id
            if job not in job_status_counts:
                job_status_counts[job] = {"total": 0, "finished": 0}
            job_status_counts[job]["total"] += 1
            if row.status == "finished":
                job_status_counts[job]["finished"] += 1

        # Schritt 2: Nur Jobs, bei denen alle Operationen 'finished' sind
        closed_jobs = [job for job, counts in job_status_counts.items() if counts["total"] == counts["finished"]]

        # Schritt 3: In Job-Tabelle status = "closed" setzen (nur passende Version)
        query = Job.update(status="closed").where(
            (Job.version == version) & (Job.job_id.in_(closed_jobs))
        )
        updated = query.execute()

        print(f"✅ {updated} Job(s) wurden auf 'closed' gesetzt (Version '{version}').")
        return closed_jobs

    @classmethod
    def clone_operations(cls, referenced_version: str, new_version: str):
        """
        Klont alle JobOperationen aus der angegebenen Referenzversion in eine neue Version.
        Die Kombination (job_id, operation) bleibt gleich, nur die Version wird ersetzt.
        Überschreibt bestehende Einträge mit (job_id, new_version, operation).
        """
        operations = cls.select().where(cls.version == referenced_version)
        count = 0

        for op in operations:
            try:
                data = {
                    "job_id": op.job_id,
                    "version": new_version,
                    "operation": op.operation,
                    "machine": op.machine,
                    "start": op.start,
                    "end": op.end,
                    "processing_time": op.processing_time,
                    "status": op.status,
                }

                cls.insert(data).on_conflict(
                    conflict_target=["job_id", "version", "operation"],
                    update=data
                ).execute()
                count += 1
            except Exception as e:
                print(f"❌ Fehler beim Klonen von Operation ({op.job_id}, {op.operation}): {e}")

        print(
            f"✅ {count} JobOperation-Einträge von Version '{referenced_version}' nach Version '{new_version}' kopiert.")



def reset_all_tables():
    """
    Löscht alle Tabellen (falls vorhanden) und erstellt sie neu.
    Achtung: Alle Daten gehen dabei verloren!
    """
    tables = [JobOperation, Job, Routing, Schedule]  # Reihenfolge beachten

    try:
        db.connect(reuse_if_open=True)
        db.drop_tables(tables, safe=True)
        db.create_tables(tables)
        print("✅ Alle Tabellen wurden erfolgreich zurückgesetzt.")
    except Exception as e:
        print(f"❌ Fehler beim Zurücksetzen der Tabellen: {e}")
    finally:
        if not db.is_closed():
            db.close()
