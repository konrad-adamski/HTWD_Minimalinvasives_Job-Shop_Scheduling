import pandas as pd
import numpy as np

# Generierung der Ankunftszeiten für geg. Job-Matrix ------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------

def generate_arrivals(df_jobs: pd.DataFrame, mean_interarrival_time: float,
                      start_time: float = 0.0, random_seed: int = 122) -> pd.DataFrame:
    # 1) Seed setzen für Reproduzierbarkeit
    np.random.seed(random_seed)

    # 2) Interarrival-Zeiten erzeugen
    jobs = df_jobs['Job'].unique().tolist()
    interarrival_times = np.random.exponential(scale=mean_interarrival_time, size=len(jobs))
    interarrival_times[0] = start_time

    # 3) Kumulieren ab start_time und auf 2 Nachkommastellen runden
    new_arrivals = np.round(start_time + np.cumsum(interarrival_times), 2)

    return pd.DataFrame({'Job': jobs, 'Arrival': new_arrivals})

    


# Berechnung der mittleren Zwischenankunftszeit für geg. Job-Matrix ---------------------------------------------------------
def calculate_mean_interarrival_time(df, u_b_mmax: float = 0.9) -> float:
    """
    Berechnet die mittlere Interarrival-Zeit t_a für ein DataFrame,
    sodass die Engpassmaschine mit Auslastung u_b_mmax (< 1.0) betrieben wird.

    Parameter:
    - df: DataFrame mit Spalten ['Job','Machine','Processing Time']
    - u_b_mmax: Ziel-Auslastung der Engpassmaschine (z.B. 0.9)

    Rückgabe:
    - t_a: mittlere Interarrival-Zeit, gerundet auf 2 Dezimalstellen
    """
    # Anzahl der unterschiedlichen Jobs
    n_jobs = df['Job'].nunique()
    # Gleichverteilung über die Jobs
    p = [1.0 / n_jobs] * n_jobs

    # Vektor der Bearbeitungszeiten auf der Engpassmaschine
    vec_t_b_mmax = _get_vec_t_b_mmax(df)

    # Berechnung der mittleren Interarrival-Zeit
    t_a = sum(p[i] * vec_t_b_mmax[i] for i in range(n_jobs)) / u_b_mmax
    return round(t_a, 2)


# Vektor (der Dauer) für die Engpassmaschine
def _get_vec_t_b_mmax(df):
    """
    Ermittelt für jeden Job die Bearbeitungszeit auf der Engpassmaschine.

    Parameter:
    - df: DataFrame mit Spalten ['Job','Machine','Processing Time']

    Rückgabe:
    - Liste der Bearbeitungszeiten auf der Engpassmaschine, in der Reihenfolge
      der ersten Vorkommen der Jobs in df['Job'].
    """
    # 1) Kopie und Machine-Spalte in int umwandeln, falls nötig
    d = df.copy()
    if d['Machine'].dtype == object:
        d['Machine'] = d['Machine'].str.lstrip('M').astype(int)

    # 2) Engpassmaschine bestimmen
    eng = _get_engpassmaschine(d)

    # 3) Job-Reihenfolge festlegen
    job_order = d['Job'].unique().tolist()

    # 4) Zeiten auf Engpassmaschine extrahieren
    proc_on_eng = d[d['Machine'] == eng].set_index('Job')['Processing Time'].to_dict()

    # 5) Vektor aufbauen (0, wenn ein Job die Maschine nicht nutzt)
    vec = [proc_on_eng.get(job, 0) for job in job_order]
    return vec

# Engpassmaschine (über die gesamten Job-Matrix)
def _get_engpassmaschine(df, debug=False):
    """
    Ermittelt die Maschine mit der höchsten Gesamtbearbeitungszeit (Bottleneck) aus einem DataFrame.

    Parameter:
    - df: DataFrame mit Spalten ['Job','Machine','Processing Time']
          Machine kann entweder als int oder als String 'M{int}' vorliegen.
    - debug: Wenn True, wird die vollständige Auswertung der Maschinenbelastung ausgegeben.

    Rückgabe:
    - Index der Engpassmaschine (int)
    """
    d = df.copy()
    # Falls Machine als 'M0','M1',... vorliegt, entfernen wir das 'M'
    if d['Machine'].dtype == object:
        d['Machine'] = d['Machine'].str.lstrip('M').astype(int)
    # Gesamtbearbeitungszeit pro Maschine
    usage = d.groupby('Machine')['Processing Time'].sum().to_dict()
    if debug:
        print("Maschinenbelastung (Gesamtverarbeitungszeit):")
        for m, total in sorted(usage.items()):
            print(f"  M{m}: {total}")
    # Maschine mit maximaler Gesamtzeit
    return max(usage, key=usage.get)
