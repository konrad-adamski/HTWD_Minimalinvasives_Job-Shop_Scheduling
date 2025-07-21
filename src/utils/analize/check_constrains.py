import pandas as pd

def is_machine_conflict_free(df_schedule: pd.DataFrame) -> bool:
    """
    Prüft, ob es Maschinenkonflikte gibt.
    Gibt True zurück, wenn konfliktfrei.
    Gibt False zurück und druckt die Konflikte, wenn Konflikte existieren.
    """
    df = df_schedule.sort_values(["Machine", "Start"]).reset_index()
    conflict_indices = []

    for machine in df["Machine"].unique():
        machine_df = df[df["Machine"] == machine].sort_values("Start")

        for i in range(1, len(machine_df)):
            prev = machine_df.iloc[i - 1]
            curr = machine_df.iloc[i]

            if curr["Start"] < prev["End"]:
                conflict_indices.extend([prev["index"], curr["index"]])

    conflict_indices = sorted(set(conflict_indices))

    if conflict_indices:
        print(f"- Maschinenkonflikte gefunden: {len(conflict_indices)} Zeilen betroffen.")
        print(df_schedule.loc[conflict_indices].sort_values(["Machine", "Start"]))
        return False
    else:
        print("+ Keine Maschinenkonflikte gefunden")
        return True


# new
def is_operation_sequence_correct(df_schedule: pd.DataFrame, job_id_column: str = "Job") -> bool:
    """
    Prüft, ob innerhalb jeder Gruppe (z.B. Job oder Production_Order_ID) die Operationen
    in der richtigen technologischen Reihenfolge ausgeführt wurden.
    Dabei wird geprüft, ob die nach Startzeit sortierten Operationen auch
    nach 'Operation' aufsteigend sind.

    Parameter:
    - df_schedule: DataFrame mit mindestens den Spalten [job_id_column, 'Operation', 'Start']
    - job_id_column: Name der Spalte, nach der gruppiert werden soll (Standard: 'Job')

    Rückgabe:
    - True, wenn alle Gruppen korrekt sortiert wurden. Sonst False mit Ausgabe betroffener Gruppen.
    """
    violations = []

    for group_id, grp in df_schedule.groupby(job_id_column):
        grp_sorted = grp.sort_values("Start")
        actual_op_sequence = grp_sorted["Operation"].tolist()
        expected_sequence = sorted(actual_op_sequence)

        if actual_op_sequence != expected_sequence:
            violations.append((group_id, actual_op_sequence))

    if not violations:
        print(f"+ Alle Gruppen wurden in korrekter Operationsreihenfolge ausgeführt.")
        return True
    else:
        print(f"- {len(violations)} Gruppe(n) mit falscher Reihenfolge nach Startzeit:")
        for group_id, seq in violations:
            print(f"  {job_id_column} {group_id}: Tatsächliche Reihenfolge: {seq}")
        return False


# new
def is_job_timing_correct(df_schedule: pd.DataFrame, job_id_column: str = "Job") -> bool:
    """
    Prüft, ob die technologischen Abhängigkeiten im Zeitplan eingehalten wurden.
    D.h. keine spätere Operation beginnt vor Ende der vorherigen in derselben Gruppe
    (z.B. Job oder Produktionsauftrag).

    Parameter:
    - df_schedule: DataFrame mit den Spalten [job_id_column, 'Operation', 'Start', 'End']
    - job_id_column: Spaltenname zur Gruppierung (Standard: 'Job')

    Rückgabe:
    - True, wenn korrekt. Sonst False mit Ausgabe der verletzenden Gruppen.
    """
    violations = []

    for group_id, grp in df_schedule.groupby(job_id_column):
        grp = grp.sort_values('Operation')  # technologisch sortieren
        previous_end = -1
        for _, row in grp.iterrows():
            if row['Start'] < previous_end:
                violations.append((group_id, int(row['Operation']), int(row['Start']), int(previous_end)))
            previous_end = row['End']

    if not violations:
        print(f"+ Alle technologischen Abhängigkeiten wurden eingehalten.")
        return True

    print(f"- {len(violations)} Verletzung(en) technologischer Abhängigkeit gefunden:")
    for group_id, op, start, prev_end in violations:
        print(f"  {job_id_column} {group_id!r}, Operation {op}: Start={start}, aber vorherige Operation endete bei {prev_end}")
    return False



def is_start_correct(df_schedule: pd.DataFrame) -> bool:
    """
    Prüft, ob alle Operationen frühestens ab Ankunftszeit ihres Produktionsplans starten.
    Erwartet, dass 'Arrival' bereits in df_schedule vorhanden ist.
    """
    violations = df_schedule[df_schedule["Start"] < df_schedule["Arrival"]]

    if violations.empty:
        print("+ Alle Operation starten erst nach Arrival des Job")
        return True
    else:
        print(f"- Fehlerhafte Starts gefunden ({len(violations)} Zeilen):")
        vals = violations.sort_values("Start")
        print(f"  {vals}")
        return False



def all_in_one(df_schedule: pd.DataFrame, job_id_column: str = "Job") -> bool:
    """
    Führt alle wichtigen Prüfungen auf einem Tages-Schedule durch:
    - Maschinenkonflikte
    - Job-Maschinen-Reihenfolge (technologische Reihenfolge)
    - Startzeiten nach technologischen Enden
    - Startzeiten nach Ankunft

    Parameter:
    - df_schedule: Zeitplan-DataFrame mit den nötigen Spalten
    - job_id_column: Spaltenname zur Gruppierung von Jobs/Aufträgen (Standard: 'Job')

    Rückgabe:
    - True, wenn alle Prüfungen bestanden wurden, sonst False.
    """

    checks_passed = True

    if not is_machine_conflict_free(df_schedule):
        checks_passed = False

    if not is_operation_sequence_correct(df_schedule, job_id_column=job_id_column):
        checks_passed = False

    if not is_job_timing_correct(df_schedule, job_id_column=job_id_column):
        checks_passed = False

    if not is_start_correct(df_schedule):
        checks_passed = False

    if checks_passed:
        print("\n+++ Alle Constraints wurden erfüllt.\n")
    else:
        print("\n--- Es wurden Constraint-Verletzungen gefunden.\n")

    return checks_passed