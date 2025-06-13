import pandas as pd
import numpy as np 

def get_times_df(df_jssp: pd.DataFrame,
                 arrivals: pd.DataFrame,
                 schedule_func,
                 target_service: float = 1.0,
                 buffer_factor: float = 1.0) -> pd.DataFrame:
    """
    Führt find_k aus und gibt ein DataFrame mit Arrival- und Deadlinedaten zurück.

    Parameter:
    - df_jssp: JSSP-Daten mit Job-Operationen
    - arrivals: DataFrame mit Spalten ['Job', 'Arrival']
    - schedule_func: Funktion zur Erstellung eines Schedules
    - target_service: Ziel-Service-Level (default: 0.95)
    - buffer_factor: zusätzlicher Puffer bei Deadlines (default: 1.75)

    Rückgabe:
    - df_times: DataFrame mit Spalten ['Job', 'Arrival', 'Deadline']
    """
    k, df_deadlines = find_k(df_jssp, arrivals, schedule_func,
                             target_service=target_service,
                             buffer_factor=buffer_factor)
    
    df_times = pd.merge(arrivals, df_deadlines, on="Job")
    df_times["Deadline"] = np.ceil(df_times["Deadline"])
                                   
    return df_times


def find_k(df_jssp: pd.DataFrame,
           arrivals: pd.DataFrame,
           schedule_func,
           target_service: float = 1.0,
           buffer_factor: float = 1.0):
    """
    Sucht per Binärsuche den Faktor k, sodass der gegebene schedule_func
    auf Basis df_jssp und arrivals einen Service-Level ≥ target_service erreicht.

    Rückgabe:
    - k: gefundener Skalierungsfaktor
    - df_deadlines: DataFrame mit Spalten ['Job','Deadline']
    """
    # Base schedule (ohne Deadlines)
    df_sched = schedule_func(df_jssp, arrivals)

    lo, hi = 0.5, 10.0
    for _ in range(30):
        k = (lo + hi) / 2
        # Deadlines als Dict für schnellen Map-Vergleich
        df_dead_tmp = _calc_due_dates(df_jssp, arrivals, k, buffer_factor=1.0)
        d_tmp = df_dead_tmp.set_index('Job')['Deadline'].to_dict()
        # Anteil pünktlicher Jobs
        on_time = (df_sched['End'] <= df_sched['Job'].map(d_tmp)).mean()
        if on_time >= target_service:
            hi = k
        else:
            lo = k

    # Finales Deadline-DF mit Puffer
    df_deadlines = _calc_due_dates(df_jssp, arrivals, k, buffer_factor)

    return k, df_deadlines


def _calc_due_dates(df_jssp: pd.DataFrame,
                    arrivals: pd.DataFrame,
                    k: float,
                    buffer_factor: float = 1.0) -> pd.DataFrame:
    """
    Berechnet Deadlines:
      d_j = a_j + (k * p_j) * buffer_factor

    Rückgabe:
    - DataFrame mit ['Job','Deadline']
    """
    # p_j: Gesamtprozesszeit
    p_tot = df_jssp.groupby('Job')['Processing Time'].sum().rename('p_tot')
    # a_j: Ankunftszeit
    a = arrivals.set_index('Job')['Arrival'].rename('a_j')
    df = pd.concat([p_tot, a], axis=1).reset_index()
    df['Deadline'] = df['a_j'] + (k * df['p_tot']) * buffer_factor
    return df[['Job', 'Deadline']]


