import pandas as pd
import numpy as np

# Ankunftszeiten ----------------------------------------------------------------------


def generate_arrivals(df_jobs: pd.DataFrame, u_b_mmax: float = 0.9, start_time: float = 0.0,
                      job_column: str = 'Job', var_type: str = 'Integer',
                      random_seed: int = 122, verbose: bool = False) -> pd.DataFrame:
    """
    Erzeugt für jeden Job in df_jobs einen Ankunftszeitpunkt basierend auf einer
    exponentialverteilten Zwischenankunftszeit, sodass die Engpassmaschine mit der
    gewünschten Auslastung u_b_mmax betrieben wird.

    Rückgabe:
    - DataFrame mit Spalten [job_column, 'Production_Plan_ID', 'Arrival']
    """
    # 1) Eindeutige Jobs extrahieren
    unique_jobs = df_jobs[[job_column, 'Production_Plan_ID']].drop_duplicates()
    
    # 2) Mittelwert berechnen
    t_a = calculate_mean_interarrival_time(df_jobs, u_b_mmax=u_b_mmax, job_column=job_column, verbose=verbose)
    if verbose:
        print(f"Mittlere Zwischenankunftszeit: {t_a}")

    # 3) Ankünfte berechnen (ausgelagert)
    arrivals = calculate_arrivals(unique_jobs, t_a, start_time=start_time,
                                  var_type=var_type, random_seed=random_seed)

    # 4) Ankunftszeiten ergänzen
    df_arrivals = unique_jobs.copy()
    df_arrivals['Arrival'] = arrivals

    return df_arrivals


def calculate_arrivals(df_jobs: pd.DataFrame, mean_interarrival_time: float, 
                       start_time: float = 0.0, var_type: str = 'Integer',
                       random_seed: int = 122) -> np.ndarray:
    """
    Berechnet eine Liste von Ankunftszeitpunkten für gegebene Jobs basierend auf
    einer exponentialverteilten Zwischenankunftszeit.

    Rückgabe:
    - Numpy-Array der Ankunftszeiten (gerundet je nach var_type)
    """
    np.random.seed(random_seed)
    n_jobs = len(df_jobs)

    interarrival_times = np.random.exponential(scale=mean_interarrival_time, size=n_jobs)
    arrivals = start_time + np.cumsum(interarrival_times)

    if var_type == 'Integer':
        arrivals = np.floor(arrivals).astype(int)
    else:
        arrivals = np.round(arrivals, 2)

    return arrivals



def calculate_mean_interarrival_time(df, u_b_mmax: float = 0.9, job_column='Job', verbose=False) -> float:
    """
    Berechnet die mittlere Interarrival-Zeit t_a für ein DataFrame,
    sodass die Engpassmaschine mit Auslastung u_b_mmax (< 1.0) betrieben wird.

    Parameter:
    - df: DataFrame mit Spalten [job_column, 'Machine', 'Processing Time']
    - u_b_mmax: Ziel-Auslastung der Engpassmaschine (z. B. 0.9)
    - job_column: Name der Spalte, die die Jobs eindeutig identifiziert
    - verbose: Wenn True, wird die Engpassermittlung und der Vektor ausgegeben

    Rückgabe:
    - t_a: mittlere Interarrival-Zeit, gerundet auf 2 Dezimalstellen
    """
    n_jobs = df[job_column].nunique()
    p = [1.0 / n_jobs] * n_jobs

    vec_t_b_mmax = _get_vec_t_b_mmax(df, job_column=job_column, verbose=verbose)

    if verbose:
        print(f"Bearbeitungszeiten auf Engpassmaschine: {vec_t_b_mmax}")

    t_a = sum(p[i] * vec_t_b_mmax[i] for i in range(n_jobs)) / u_b_mmax
    return round(t_a, 2)


def _get_vec_t_b_mmax(df, job_column='Job', verbose=False):
    """
    Gibt die Bearbeitungszeit jedes Jobs auf der Engpassmaschine zurück,
    in der Reihenfolge des ersten Auftretens in df[job_column].

    Parameter:
    - df: DataFrame mit Spalten [job_column, 'Machine', 'Processing Time']
    - job_column: Name der Spalte, die den Job oder Auftrag eindeutig identifiziert.

    Rückgabe:
    - Liste der Bearbeitungszeiten auf der Engpassmaschine pro Job (0, wenn der Job die Maschine nicht nutzt).
    """
    eng = _get_engpassmaschine(df, verbose=verbose)
    job_order = df[job_column].unique().tolist()
    proc_on_eng = df[df['Machine'] == eng].set_index(job_column)['Processing Time'].to_dict()
    return [proc_on_eng.get(job, 0) for job in job_order]


def _get_engpassmaschine(df, verbose=False):
    """
    Ermittelt die Maschine mit der höchsten Gesamtbearbeitungszeit (Engpassmaschine).

    Parameter:
    - df: DataFrame mit Spalten ['Machine', 'Processing Time']
          'Machine' kann beliebige Label enthalten (z. B. int, str).
    - verbose: Wenn True, wird die Maschinenbelastung ausgegeben.

    Rückgabe:
    - Bezeichnung der Engpassmaschine (gleicher Typ wie in der Spalte 'Machine').
    """
    usage = df.groupby('Machine')['Processing Time'].sum().to_dict()
    if verbose:
        print("Maschinenbelastung (Gesamtverarbeitungszeit):")
        for m, total in sorted(usage.items(), key=lambda x: str(x[0])):
            print(f"  {m}: {total}")
    return max(usage, key=usage.get)

