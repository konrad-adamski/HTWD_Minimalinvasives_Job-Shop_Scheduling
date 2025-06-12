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
def is_operation_sequence_correct(df_schedule: pd.DataFrame) -> bool:
    """
    Prüft, ob innerhalb jedes Jobs die Operationen in der richtigen technologischen Reihenfolge
    ausgeführt wurden. Dabei wird geprüft, ob die nach Startzeit sortierten Operationen
    auch nach 'Operation' aufsteigend sind.

    Rückgabe:
    - True, wenn korrekt. Sonst False mit Ausgabe betroffener Jobs.
    """
    violations = []

    for job, grp in df_schedule.groupby("Job"):
        grp_sorted = grp.sort_values("Start")
        actual_op_sequence = grp_sorted["Operation"].tolist()
        expected_sequence = sorted(actual_op_sequence)

        if actual_op_sequence != expected_sequence:
            violations.append((job, actual_op_sequence))

    if not violations:
        print("+ Alle Jobs wurden in korrekter Operationsreihenfolge ausgeführt.")
        return True
    else:
        print(f"- {len(violations)} Job(s) mit falscher Reihenfolge nach Startzeit:")
        for job, seq in violations:
            print(f"  Job {job}: Tatsächliche Reihenfolge: {seq}")
        return False


# new
def is_job_timing_correct(df_schedule: pd.DataFrame) -> bool:
    """
    Prüft, ob die technologischen Abhängigkeiten im Zeitplan eingehalten wurden.
    D.h. keine spätere Operation beginnt vor Ende der vorherigen im selben Job.

    Rückgabe:
    - True, wenn korrekt. Sonst False mit Ausgabe der verletzenden Jobs.
    """
    violations = []

    for job, grp in df_schedule.groupby('Job'):
        grp = grp.sort_values('Operation')  # technologisch korrekt sortieren
        previous_end = -1
        for _, row in grp.iterrows():
            if row['Start'] < previous_end:
                violations.append((job, int(row['Operation']), int(row['Start']), int(previous_end)))
            previous_end = row['End']

    if not violations:
        print("+ Alle technologischen Abhängigkeiten sind eingehalten.")
        return True

    print(f"- {len(violations)} Verletzung(en) der technologischen Reihenfolge gefunden:")
    for job, op, start, prev_end in violations:
        print(f"  Job {job!r}, Operation {op}: Start={start}, aber vorherige Operation endete erst bei {prev_end}")
    return False



def is_start_correct(df_schedule: pd.DataFrame) -> bool:
    """
    Prüft, ob alle Operationen frühestens ab ihrer Ankunftszeit starten.
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



def check_constraints(df_schedule: pd.DataFrame) -> bool:
    """
    Führt alle wichtigen Prüfungen auf einem Tages-Schedule durch:
    - Maschinenkonflikte
    - Job-Maschinen-Reihenfolge
    - Startzeiten nach Ankunft
    Gibt True zurück, wenn alle Prüfungen bestanden sind, sonst False.
    """

    checks_passed = True

    if not is_machine_conflict_free(df_schedule):
        checks_passed = False

    if not is_operation_sequence_correct(df_schedule):
        checks_passed = False

    if not is_job_timing_correct(df_schedule):
        checks_passed = False

    if not is_start_correct(df_schedule):
        checks_passed = False

    if checks_passed:
        print("\n+++ Alle Constraints wurden erfüllt.\n")
    else:
        print("\n--- Es wurden Constraint-Verletzungen gefunden.\n")

    return checks_passed