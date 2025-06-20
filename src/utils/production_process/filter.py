import pandas as pd


def jobs_by_ready_time(df_jobs: pd.DataFrame, df_ops: pd.DataFrame, 
                              ready_time_col = "Ready Time", ready_time: int = 0, verbose = False) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Jobs zeitlich filtern
    time_filter = df_jobs[ready_time_col] == ready_time
    df_jobs_filtered = df_jobs[time_filter].copy()

    # Operationen nach (gefilterten) Jobs filtern
    jobs = df_jobs_filtered["Job"].unique()
    df_ops_filtered = df_ops[df_ops["Job"].isin(jobs)].copy()

    if verbose:
        print(f"[INFO] Anzahl Jobs mit {ready_time_col} {ready_time}: {len(jobs)}")
        
    return df_ops_filtered,df_jobs_filtered



# I) Init Filtern nach Teitfenster -------------------------------------------------------------------
def jobs_by_arrival_window(df_times: pd.DataFrame, df_jssp: pd.DataFrame,
                           day_start: float = 0, planning_end: float | None = None,
                           job_column = 'Job', arrival_column: str = 'Arrival', verbose = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filtert Jobs anhand eines Zeitfensters (z.B. Tagesabschnitt) und gibt die passenden
    Datensätze für Ankunftszeiten und JSSP zurück.

    Parameter:
    - df_times: DataFrame mit [job_column', arrival_column], z.B. Ankunftszeiten.
    - df_jssp: DataFrame mit [job_column, 'Operation', 'Machine', 'Processing Time'].
    - day_start: Startzeit des Zeitfensters.
    - planning_end: Endzeit des Zeitfensters.
    - job_column: Name der Spalte mit den Produktionaufträgen (Standard: 'Job').
    - arrival_column: Name der Spalte mit den Ankunftszeiten (Standard: 'Arrival').

    Rückgabe:
    - df_times_filtered: Nur Jobs, deren Arrival im Fenster liegt.
    - df_jssp_filtered: Entsprechende Operationen aus df_jssp.
    """

    # Einheitliche Typkonvertierung
    df_times[job_column] = df_times[job_column].astype(str)
    df_jssp[job_column] = df_jssp[job_column].astype(str)

    time_filter = df_times[arrival_column] >= day_start
    if planning_end is not None:
        time_filter &= df_times[arrival_column] < planning_end
        
    df_times_filtered = df_times[time_filter].copy()
    
    relevant_jobs = df_times_filtered[job_column].unique()
    df_jssp_filtered = df_jssp[df_jssp[job_column].isin(relevant_jobs)].copy()

    if verbose:
        print(f"[INFO] Jobs zwischen {day_start} und {planning_end}: {len(relevant_jobs)} Jobs gefunden.")
         
    return df_jssp_filtered, df_times_filtered


#  After Sim -----------------------------------------------------------------------------------------------------------------
def get_unexecuted_operations(df_plan: pd.DataFrame, df_execution: pd.DataFrame, job_column="Job") -> pd.DataFrame:
    """
    Gibt alle Operationen aus df_plan zurück, die noch nicht in df_execution enthalten sind.
    Nutzt einen Anti-Join auf [job_column, 'Operation'].

    Parameter:
    - df_plan: DataFrame mit allen geplanten Operationen. Muss mindestens die Spalten [job_column, 'Operation'] enthalten.
    - df_execution: DataFrame mit allen bereits ausgeführten Operationen. Muss ebenfalls [job_column, 'Operation'] enthalten.
    - job_column: Spaltenname, der die Job-ID bezeichnet (Standard: 'Job').

    Rückgabe:
    - DataFrame mit Operationen aus df_plan, die noch nicht in df_execution enthalten sind.
    """
    # Einheitliche Typkonvertierung
    df_plan[job_column] = df_plan[job_column].astype(str)
    df_execution[job_column] = df_execution[job_column].astype(str)
    
    executed_keys = df_execution[[job_column, 'Operation']].drop_duplicates()

    merged = df_plan.merge(
        executed_keys,                         # Nur die relevanten Schlüsselspalten
        on=[job_column, 'Operation'],          # Vergleich anhand von Job und Operation
        how='left',                            # Behalte alle Zeilen aus df_plan
        indicator=True                         # Erzeugt die Spalte '_merge' für Join-Analyse
    )

    # Behalte nur die Zeilen, bei denen es keinen Match gab – also noch nicht ausgeführte Operationen
    df_plan_undone = merged[merged['_merge'] == 'left_only']

    # Entferne die Hilfsspalte '_merge' und setze den Index zurück
    df_plan_undone = df_plan_undone.drop(columns=['_merge']).reset_index(drop=True)

    # Rückgabe des gefilterten DataFrames
    return df_plan_undone


def get_operations_running_into_day(df_execution: pd.DataFrame, day_start: float, verbose: bool = False) -> pd.DataFrame:
    """
    Gibt alle Operationen zurück, deren Endzeit in oder nach dem gegebenen Tagesstart liegt.
    D.h. alle Operationen, die noch aktiv sind oder über den Tageswechsel hinauslaufen.

    Parameter:
    - df_execution: DataFrame mit mindestens der Spalte 'End'.
    - day_start: Startzeit des betrachteten Tages (z.B. 1440.0 für Tag 2 bei Minutenmodellierung).
    - verbose: Wenn True, wird ausgegeben, wie viele Operationen erst heute enden.

    Rückgabe:
    - DataFrame mit relevanten Operationen.
    """
    df_filtered = df_execution[df_execution["End"] >= day_start].copy()

    if verbose:
        count = len(df_filtered)
        print(f"[INFO] {count} laufende Operation(en) aus vorherigen Tagen enden erst nach Tagesbeginn.")

    return df_filtered


def extend_with_undone_operations(df_jssp_todo: pd.DataFrame, df_undone: pd.DataFrame, job_column ="Job", verbose: bool = False) -> pd.DataFrame:
    """
    Kombiniert noch nicht gestartete Operationen mit abgebrochenen Operationen.
    Achtet auf Einheitlichkeit der Datentypen und entfernt Duplikate.

    Parameter:
    - df_jssp_todo: DataFrame mit geplanten, aber noch nicht ausgeführten Operationen.
    - df_undone: DataFrame mit während des Tages abgebrochenen Operationen.

    Rückgabe:
    - df_jssp_todo_extended: Kombinierter, bereinigter DataFrame.
    """

    # Einheitlicher Datentyp für 'Job'
    df_undone[job_column] = df_undone[job_column].astype(str)
    df_jssp_todo[job_column] = df_jssp_todo[job_column].astype(str)

    # Anzahl vor dem Hinzufügen merken
    original_count = len(df_jssp_todo)
    
    # Relevante Spalten aus df_undone
    base_cols = [job_column, 'Operation', 'Machine', 'Processing Time']
    if 'Production_Plan_ID' in df_undone.columns:
        base_cols = [job_column, 'Production_Plan_ID', 'Operation', 'Machine', 'Processing Time']
    df_undone_relevant = df_undone[base_cols].copy()

    # Explizite Kopie von df_jssp_todo zur sicheren Bearbeitung
    df_jssp_todo_copy = df_jssp_todo.copy()

    # Kombination beider DataFrames
    df_combined = pd.concat([df_undone_relevant, df_jssp_todo_copy], ignore_index=True)

    # Doppelte Operationen entfernen
    df_combined.drop_duplicates(subset=[job_column, 'Operation'], inplace=True)

    # Index neu setzen
    df_combined.reset_index(drop=True, inplace=True)

    # Ausgabe, wie viele neue Zeilen hinzugefügt wurden
    if verbose:
        new_count = len(df_combined)
        added = new_count - original_count
        print(f"[INFO] {added} zusätzliche Operationen hinzugefügt (gesamt: {new_count}).")

    return df_combined


def erase_done_operations(df_jssp_todo: pd.DataFrame, df_done: pd.DataFrame, job_column ="Job", verbose: bool = False) -> pd.DataFrame:
    """
    Entfernt aus df_jssp_todo alle Operationen, die in df_done enthalten sind.

    Parameter:
    - df_jssp_todo: DataFrame mit geplanten oder verbleibenden Operationen.
    - df_done: DataFrame mit bereits erledigten Operationen.

    Rückgabe:
    - df_filtered: df_jssp_todo ohne die erledigten Operationen.
    """

    # Einheitlicher Typ für Job
    df_jssp_todo[job_column] = df_jssp_todo[job_column].astype(str)
    df_done[job_column] = df_done[job_column].astype(str)
    
    # Anzahl vor dem Entfernen merken
    original_count = len(df_jssp_todo)
    
    # Kopie für sichere Bearbeitung
    df_jssp_todo_copy = df_jssp_todo.copy()

    df_done_keys = df_done[[job_column, 'Operation']].astype(str).drop_duplicates()

    # Entferne erledigte Operationen
    df_filtered = df_jssp_todo_copy.merge(
        df_done_keys, on=[job_column, 'Operation'], how='left', indicator=True
    )
    df_filtered = df_filtered[df_filtered['_merge'] == 'left_only']
    df_filtered.drop(columns=['_merge'], inplace=True)

    df_filtered.reset_index(drop=True, inplace=True)
    
    # Ausgabe, wie viele Operationen entfernt wurden
    if verbose:
        removed = original_count - len(df_filtered)
        print(f"[INFO] {removed} Operationen entfernt (verbleibend: {len(df_filtered)}).")
        
    return df_filtered


def update_times_after_operation_changes(df_times: pd.DataFrame, df_jssp_todo: pd.DataFrame, job_column: str = "Job") -> pd.DataFrame:
    """
    Aktualisiert df_times basierend auf dem aktuellen Stand von df_jssp_todo.
    Entfernt veraltete Zeiteinträge und ergänzt ggf. fehlende, indem nur Jobs
    berücksichtigt werden, die tatsächlich noch geplante Operationen haben.

    Dies ist notwendig, wenn im Planungsprozess Operationen entfernt oder hinzugefügt wurden.

    Parameter:
    - df_times: Ursprünglicher DataFrame mit Zeiteinträgen (z.B. Ankunftszeiten).
    - df_jssp_todo: Aktueller Satz geplanter Operationen.
    - job_column: Name der Spalte mit der Job-ID (Standard: 'Job').

    Rückgabe:
    - df_times_updated: Bereinigter df_times mit nur noch relevanten Jobs.
    """
    # Einheitlicher Typ für Job
    df_times[job_column] = df_times[job_column].astype(str)
    df_jssp_todo[job_column] = df_jssp_todo[job_column].astype(str)
    
    relevant_jobs = df_jssp_todo[job_column].unique()
    df_times_updated = df_times[df_times[job_column].isin(relevant_jobs)].copy()
    return df_times_updated.reset_index(drop=True)    
