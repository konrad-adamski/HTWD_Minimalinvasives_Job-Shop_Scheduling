from src.utils.initialization.arrivals_init import calculate_mean_interarrival_time, generate_arrivals_from_mean_interarrival_time
import pandas as pd
import numpy as np
import random

def create_jobs_for_shifts(df_routings: pd.DataFrame, routing_column: str = 'Routing_ID', job_column: str = 'Job',
                           shift_count: int = 1, shift_length: int = 1440, u_b_mmax: float = 0.9,
                           shuffle: bool = False, job_seed: int = 50, arrival_seed: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Erzeugt Jobs für mehrere Schichten und berechnet die Ankunftszeiten
    auf Basis einer Zielauslastung.

    Parameter:
    - df_routings: Vorlage mit [routing_column, 'Operation', 'Machine', 'Processing Time']
    - job_column: Name der Spalte für Job-IDs
    - routing_column: Name der Spalte für Routing-IDs
    - shift_count: Anzahl der Schichten
    - u_b_mmax: Zielauslastung der Engpassmaschine
    - shift_length: Dauer einer Schicht in Minuten
    - shuffle: Ob Routings innerhalb der Wiederholungen gemischt werden sollen
    - job_seed: Zufallsseed für Job-Generierung
    - arrival_seed: Zufallsseed für Ankunftszeiten

    Rückgabe:
    - df_jssp: Jobs mit Operationen
    - df_arrivals: Zeitplan mit [job_column, routing_column, Arrival]
    """
    # 1) Jobs generieren (mehrere Wiederholungen)
    multiplicator = 2 + shift_length // 500
    repetitions = multiplicator * shift_count
    df_jssp = generate_multiple_jobs_from_routings(df_routings=df_routings, job_column=job_column,
                                                   routing_column=routing_column, repetitions=repetitions, 
                                                   shuffle=shuffle, seed=job_seed
                                                  )

    # 2) Mittlere Zwischenankunftszeit bestimmen
    t_a = calculate_mean_interarrival_time(df_routings, u_b_mmax = u_b_mmax, routing_column= routing_column)

    # 3) Ankunftszeiten
    unique_jobs = df_jssp[[job_column, routing_column]].drop_duplicates()
    job_numb = len(unique_jobs)
    arrivals = generate_arrivals_from_mean_interarrival_time(job_numb, t_a, var_type="Integer", random_seed=arrival_seed)

    df_jobs_arrivals = unique_jobs.copy()
    df_jobs_arrivals['Arrival'] = arrivals

    # 4) a) Nur Ankunftszeiten, die im Zeitfenster sind, behalten
    time_limit = shift_count * shift_length
    df_jobs_arrivals = df_jobs_arrivals[df_jobs_arrivals['Arrival'] < time_limit].reset_index(drop=True)

    # 4) b) Dazugehörige Produktionsaufträge behalten
    valid_ids = set(df_jobs_arrivals[job_column])
    df_jssp = df_jssp[df_jssp[job_column].isin(valid_ids)].reset_index(drop=True)


    # 5) Ready Time
    df_jobs_arrivals['Ready Time'] = np.ceil((df_jobs_arrivals['Arrival'] + 1) / shift_length) * shift_length
    df_jobs_arrivals['Ready Time'] = df_jobs_arrivals['Ready Time'].astype(int)

    return df_jssp, df_jobs_arrivals
    


def generate_multiple_jobs_from_routings(df_routings: pd.DataFrame, job_column: str = 'Job', routing_column: str = 'Routing_ID',
                                         repetitions: int = 3, shuffle: bool = False, seed: int = 50) -> pd.DataFrame:
    """
    Ruft `generate_jobs_from_routings` mehrfach auf und erzeugt eine kombinierte Menge neuer Jobs
    mit fortlaufenden Job-IDs.

    Parameter:
    - df_routings: Vorlage mit [routing_column, 'Operation', 'Machine', 'Processing Time']
    - repetitions: Anzahl der Wiederholungen
    - shuffle: Ob die Gruppen bei jedem Durchlauf gemischt werden sollen
    - seed: Basis-Zufallsseed 

    Rückgabe:
    - Kombinierter DataFrame mit [job_column, routing_column, 'Operation', 'Machine', 'Processing Time']
    """
    all_jobs = []
    routings_per_repetition = df_routings[routing_column].nunique()

    for i in range(repetitions):
        offset = i * routings_per_repetition
        current_seed = seed + i
        curr_shuffle = shuffle and (i % 100 != 1)  # Alle 50. Wiederholungen (i = 1, 101, 201, ...) nicht shuffeln
        
        df_jobs = generate_jobs_from_routings(
            df_routings,
            job_column=job_column,
            routing_column=routing_column,
            offset=offset,
            shuffle=curr_shuffle,
            seed=current_seed
        )
        all_jobs.append(df_jobs)

    return pd.concat(all_jobs, ignore_index=True)



def generate_jobs_from_routings(df_routings: pd.DataFrame, job_column: str = 'Job', routing_column: str = 'Routing_ID',
                                offset: int = 0, shuffle: bool = False, seed: int = 50) -> pd.DataFrame:
    """
    Erzeugt neue Jobs aus dem gegebenen Routing-Template mit fortlaufenden Job-IDs.

    Jeder Job basiert auf einem Arbeitsplan (routing_column).

    Parameter:
    - df_routings: DataFrame mit [routing_column, 'Operation', 'Machine', 'Processing Time']
    - offset: Startindex für neue Job-IDs (z.B. 0 für erste Job-ID)
    - shuffle: Ob die Reihenfolge der Routing-Vorlagen gemischt werden soll
    - seed: Zufalls-Seed für das Mischen

    Rückgabe:
    - DataFrame mit [job_column, routing_column, 'Operation', 'Machine', 'Processing Time']
    """

    # 1) Routing-Vorlagen gruppieren
    groups = [grp for _, grp in df_routings.groupby(routing_column, sort=False)]

    # 2) Optional mischen
    if shuffle:
        random.seed(seed)
        random.shuffle(groups)

    # 3) Neue Jobs erzeugen
    new_recs = []
    for i, grp in enumerate(groups):
        job_id = offset + i
        routing_id = grp[routing_column].iloc[0]
        for _, row in grp.iterrows():
            new_recs.append({
                job_column: f"J25-{job_id:04d}",
                routing_column: routing_id,
                'Operation': row['Operation'],
                'Machine': row['Machine'],
                'Processing Time': row['Processing Time']
            })

    return pd.DataFrame(new_recs).reset_index(drop=True)