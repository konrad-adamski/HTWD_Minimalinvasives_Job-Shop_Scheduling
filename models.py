import pandas as pd
from peewee import Model, TextField, CharField, ForeignKeyField, IntegerField, AutoField, BooleanField
from playhouse.sqlite_ext import JSONField

from database import db


class RoutingDefinition(Model):
    routing_id = CharField(primary_key=True, verbose_name="Routing ID")
    description = TextField(null=True, verbose_name="Description")

    class Meta:
        database = db

class Routing(Model):
    routing = ForeignKeyField(RoutingDefinition, backref='operations', on_delete='CASCADE', verbose_name="Routing ID")
    operation = IntegerField(verbose_name="Operation")
    machine = CharField(verbose_name="Machine")
    duration = IntegerField(verbose_name="Duration")

    class Meta:
        database = db
        indexes = (
            (('routing', 'operation'), True),
        )

class Job(Model):
    job_id = IntegerField(primary_key=True, verbose_name="Job ID")
    routing = ForeignKeyField(RoutingDefinition, backref='jobs', on_delete='CASCADE', verbose_name="Routing ID")
    arrival_time = IntegerField(verbose_name="Arrival Time")
    ready_time = IntegerField(verbose_name="Ready Time")
    due_date = IntegerField(verbose_name="Due Date")
    status = BooleanField(default=False, verbose_name="Status")

    class Meta:
        database = db

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, routing_column = 'Routing_ID', due_date_column = 'Deadline'):
        """
        Erstellt Job-Objekte aus einem DataFrame und speichert sie in der Datenbank.
        Erwartet Spalten: 'Job', 'Routing_ID', 'Arrival', 'Ready Time', 'Deadline'
        """
        for _, row in df.iterrows():
            try:
                cls.create(
                    job_id=int(row['Job']),
                    routing=RoutingDefinition.get_by_id(int(row[routing_column])),
                    arrival_time=int(row['Arrival']),
                    ready_time=int(row['Ready Time']),
                    due_date=int(row[due_date_column]),
                    status=False
                )
            except RoutingDefinition.DoesNotExist:
                print(f"⚠️ Routing_ID {row['Routing_ID']} nicht gefunden – Job {row['Job']} wird übersprungen.")
            except Exception as e:
                print(f"❌ Fehler bei Job {row['Job']}: {e}")

class Schedule(Model):
    id = AutoField(verbose_name="ID")
    description = TextField(null=True, verbose_name="Description")
    data = JSONField(verbose_name="Data")

    class Meta:
        database = db