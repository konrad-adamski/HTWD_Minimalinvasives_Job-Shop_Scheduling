from src.utils.production_process import time_determination as term
import pandas as pd
import random

def create_production_orders_for_shifts(df_template: pd.DataFrame, job_column: str ='Job', shift_count: int = 1, u_b_mmax: float = 0.9,
                                        shift_length: int = 1440, shuffle: bool = False, job_seed: int = 50,
                                        arrival_seed: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Erzeugt Produktionsaufträge für mehrere Schichten und berechnet die Ankunftszeiten
    auf Basis einer Zielauslastung.

    Parameter:
    - df_template: Vorlage mit ['Production_Plan_ID', 'Operation', 'Machine', 'Processing Time']
    - shift_count: Anzahl der Schichten
    - u_b_mmax: Zielauslastung der Engpassmaschine
    - shift_length: Dauer einer Schicht in Minuten
    - shuffle: Ob Prozesspläne innerhalb der Wiederholungen gemischt werden sollen
    - job_seed: Zufallsseed für Auftragsgenerierung
    - arrival_seed: Zufallsseed für Ankunftszeiten

    Rückgabe:
    - df_jssp: Produktionsaufträge mit Operationen
    - df_arrivals: Zeitplan mit [job_column, Production_Plan_ID, Arrival]
    """
    # 1) Aufträge generieren (mehrere Wiederholungen)
    multiplicator = 2 + shift_length // 500
    repetitions = multiplicator * shift_count
    df_jssp = create_multiple_production_orders(df_template=df_template, job_column=job_column, repetitions=repetitions, shuffle=shuffle, seed=job_seed)

    # 2) Ankunftszeiten berechnen
    df_arrivals = term.generate_arrivals(df_jobs=df_jssp, u_b_mmax=u_b_mmax, start_time=0.0, 
                                    job_column=job_column, var_type='Integer', random_seed=arrival_seed)

    # 3) Nur Aufträge im Zeitfenster behalten
    time_limit = shift_count * shift_length
    df_arrivals = df_arrivals[df_arrivals['Arrival'] < time_limit].reset_index(drop=True)

    # 4) Nur gültige Aufträge behalten
    valid_ids = set(df_arrivals[job_column])
    df_jssp = df_jssp[df_jssp[job_column].isin(valid_ids)].reset_index(drop=True)

    return df_jssp, df_arrivals




def create_multiple_production_orders(df_template: pd.DataFrame, job_column: str ='Job', repetitions: int = 3,
                                      shuffle: bool = False, seed: int = 50) -> pd.DataFrame:
    """
    Ruft `production_orders` mehrfach auf und erzeugt eine kombinierte Menge neuer Aufträge
    mit fortlaufenden Produktionsauftrags-IDs.

    Parameter:
    - df_template: Vorlage mit ['Production_Plan_ID', 'Operation', 'Machine', 'Processing Time']
    - repetitions: Anzahl der Wiederholungen
    - shuffle: Ob die Gruppen bei jedem Durchlauf gemischt werden sollen
    - seed: Basis-Zufallsseed 

    Rückgabe:
    - Kombinierter DataFrame mit [job_column, 'Production_Plan_ID', 'Operation', 'Machine', 'Processing Time']
    """
    all_orders = []
    plans_per_repetition = df_template['Production_Plan_ID'].nunique()

    for i in range(repetitions):
        offset = i * plans_per_repetition
        current_seed = seed + i
        curr_shuffle = shuffle and (i % 30 != 0) # Alle 30. Wiederholungen (i = 0, 30, 60, ...) nicht shuffeln
        
        df_orders = production_orders(
            df_template,
            job_column=job_column,
            offset=offset,
            shuffle=curr_shuffle,
            seed=current_seed
        )
        all_orders.append(df_orders)

    return pd.concat(all_orders, ignore_index=True)


def production_orders(df_template: pd.DataFrame, job_column: str ='Job', 
                      offset: int = 0, shuffle: bool = False, seed: int = 50) -> pd.DataFrame:
    """
    Erzeugt neue Produktionsaufträge aus dem gegebenen Template mit fortlaufenden IDs.

    Jeder Produktionsauftrag basiert auf einem Arbeitsplan (Production_Plan_ID).

    Parameter:
    - df_template: DataFrame mit ['Production_Plan_ID', 'Operation', 'Machine', 'Processing Time']
    - offset: Startindex für neue Job IDs (z.B. 0 für erste Order-ID)
    - shuffle: Ob die Reihenfolge der Vorlagen-Pläne gemischt werden soll
    - seed: Zufalls-Seed für das Mischen

    Rückgabe:
    - DataFrame mit [job_column, 'Production_Plan_ID', 'Operation', 'Machine', 'Processing Time']
    """

    # 1) Vorlage-Pläne gruppieren
    groups = [grp for _, grp in df_template.groupby('Production_Plan_ID', sort=False)]

    # 2) Optional mischen
    if shuffle:
        random.seed(seed)
        random.shuffle(groups)

    # 3) Neue Produktionsaufträge erzeugen
    new_recs = []
    for i, grp in enumerate(groups):
        job_id = offset + i
        production_plan_id = grp['Production_Plan_ID'].iloc[0]
        for _, row in grp.iterrows():
            new_recs.append({
                job_column: job_id,
                'Production_Plan_ID': production_plan_id,
                'Operation': row['Operation'],
                'Machine': row['Machine'],
                'Processing Time': row['Processing Time']
            })

    return pd.DataFrame(new_recs).reset_index(drop=True)