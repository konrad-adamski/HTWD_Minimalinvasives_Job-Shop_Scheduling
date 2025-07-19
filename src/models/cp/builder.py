
def get_records_from_cp(
        jobs, all_ops, starts, solver,
        job_column="Job", df_times=None):
    """
    Erstellt Scheduling-Records mit Tardiness- und Earliness-Berechnung aus CP-Solver-Ergebnissen.
    Nutzt dynamische Zeitdaten aus df_times, z.B. Routing_ID, Arrival, Ready Time, Deadline (sofern vorhanden).

    Parameter:
    - jobs: Liste der Job-IDs.
    - all_ops: Liste der Operationen je Job [(op_id, machine, duration), ...].
    - starts: Dict mit CP-Startvariablen (solver.Value(var)).
    - solver: cp_model.CpSolver().
    - job_column: Name der Jobspalte im DataFrame.
    - df_times: Optionaler DataFrame mit Zusatzinfos je Job.

    Rückgabe:
    - Liste von Dicts mit allen Scheduling-Informationen pro Operation.
    """

    # 1. Dynamisch relevante Spalten aus df_times extrahieren
    relevant_columns = ["Routing_ID", "Arrival", "Ready Time", "Deadline"]
    times_dict = {}

    if df_times is not None:
        df_times = df_times.set_index(job_column)
        for col in relevant_columns:
            if col in df_times.columns:
                times_dict[col] = df_times[col].to_dict()

    # 2. Records aufbauen
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = solver.Value(starts[(j, o)])
            ed = st + d

            record = {job_column: job}

            # Falls vorhanden, Zusatzinfos ergänzen
            for col, mapping in times_dict.items():
                if job in mapping:
                    record[col] = mapping[job]

            # Basisinformationen ergänzen
            record.update({
                "Operation": op_id,
                "Machine": m,
                "Start": st,
                "Processing Time": d,
                "End": ed,
            })

            records.append(record)

    return records