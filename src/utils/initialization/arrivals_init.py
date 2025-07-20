import pandas as pd
import numpy as np


def generate_arrivals_from_mean_interarrival_time(job_number: int,
                                                  mean_interarrival_time: float,
                                                  start_time: float = 0.0,
                                                  var_type: str = 'Integer',
                                                  random_seed: int = 122) -> np.ndarray:
    """
    Berechnet eine Liste von Ankunftszeitpunkten auf Basis einer mittleren Zwischenankunftszeit.

    Parameter:
    - job_number: Anzahl der zu erzeugenden Ankünfte
    - mean_interarrival_time: Erwartungswert der Exponentialverteilung
    - start_time: Startzeitpunkt für die erste Ankunft
    - var_type: 'Integer' für ganze Minuten, sonst Fließkommawerte mit 2 Nachkommastellen
    - random_seed: Seed zur Reproduzierbarkeit

    Rückgabe:
    - Numpy-Array der Ankunftszeiten (gerundet je nach var_type)
    """
    np.random.seed(random_seed)

    # 1) Exponentiell verteilte Zwischenankunftszeiten
    interarrival_times = np.random.exponential(scale=mean_interarrival_time, size=job_number)

    # 2) Kumulative Ankunftszeiten berechnen
    arrivals = start_time + np.cumsum(interarrival_times)

    # 3) Runden je nach Typ
    if var_type == 'Integer':
        arrivals = np.floor(arrivals).astype(int)
    else:
        arrivals = np.round(arrivals, 2)

    return arrivals



def calculate_mean_interarrival_time(df, u_b_mmax: float = 0.9, routing_column='Routing_ID', verbose=False) -> float:
    """
    Berechnet die mittlere Interarrival-Zeit t_a für ein DataFrame,
    sodass die Engpassmaschine mit Auslastung u_b_mmax (< 1.0) betrieben wird.

    Parameter:
    - df: DataFrame mit Spalten [routing_column, 'Machine', 'Processing Time']
    - u_b_mmax: Ziel-Auslastung der Engpassmaschine (z. B. 0.9)
    - routing_column: Name der Spalte, die die Routings eindeutig identifiziert
    - verbose: Wenn True, wird die Engpassermittlung und der Vektor ausgegeben

    Rückgabe:
    - t_a: mittlere Interarrival-Zeit, gerundet auf 2 Dezimalstellen
    """
    n_routings = df[routing_column].nunique()
    p = [1.0 / n_routings] * n_routings

    vec_t_b_mmax = _get_vec_t_b_mmax(df, routing_column=routing_column, verbose=verbose)

    if verbose:
        print(f"Bearbeitungszeiten auf Engpassmaschine: {vec_t_b_mmax}")

    t_a = sum(p[i] * vec_t_b_mmax[i] for i in range(n_routings)) / u_b_mmax
    return round(t_a, 2)


def _get_vec_t_b_mmax(df, routing_column='Routing_ID', verbose=False):
    """
    Gibt die Bearbeitungszeit jedes Routings auf der Engpassmaschine zurück,
    in der Reihenfolge des ersten Auftretens in df[routing_column].

    Parameter:
    - df: DataFrame mit Spalten [routing_column, 'Machine', 'Processing Time']
    - routing_column: Name der Spalte, die den Routing oder Auftrag eindeutig identifiziert.

    Rückgabe:
    - Liste der Bearbeitungszeiten auf der Engpassmaschine pro Routing (0, wenn der Routing die Maschine nicht nutzt).
    """
    eng = _get_engpassmaschine(df, verbose=verbose)
    routing_order = df[routing_column].unique().tolist()
    proc_on_eng = df[df['Machine'] == eng].set_index(routing_column)['Processing Time'].to_dict()
    return [proc_on_eng.get(routing, 0) for routing in routing_order]


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

