import src.utils.gen_interarrival as gen_interarrival 
import pandas as pd
import random

def filter_ops_and_jobs_by_ready_time(df_jobs: pd.DataFrame, df_ops: pd.DataFrame, 
                              ready_time_col = "Ready Time", ready_time: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Jobs zeitlich filtern
    time_filter = df_jobs[ready_time_col] == ready_time
    df_jobs_filtered = df_jobs[time_filter].copy()

    # Operationen nach (gefilterten) Jobs filtern
    jobs = df_jobs_filtered["Job"]
    df_ops_filtered = df_ops[df_ops["Job"].isin(jobs)].copy()
    return df_jobs_filtered, df_ops_filtered


def filter_plan_for_today(df_plan, latest_op_start: int = 0): # exclusive
    filt = (df_plan.Start < latest_op_start)
    return df_plan[filt].sort_values(by="Job").reset_index(drop=True)



def filter_plan_for_future(df_plan, earliest_op_start: int = 0):
    filt = (df_plan.Start >= earliest_op_start)
    return df_plan[filt].sort_values(by=["Job", "Start"]).reset_index(drop=True)


# neu
def create_jobs_for_shifts(df_template: pd.DataFrame, shift_count: int = 1, u_b_mmax: float = 0.9, 
                                  shift_length: int = 1440, shuffle: bool = False, job_seed: int = 50, arrival_seed: int = 120) -> pd.DataFrame:

    # 1) Aufträge generieren
    repetitions = 4 * shift_count
    df_jssp = create_multiple_jobs(df_template, repetitions=repetitions, shuffle=shuffle, seed=job_seed)


    # 2) Zeit-Informationen erzeugen (ab Tag 0)
    df_jobs_times = gen_interarrival.generate_job_times(df_jssp, start_time=0.0, u_b_mmax=u_b_mmax, shift_length = shift_length, seed = arrival_seed)
    

    # 3) Nur Jobs behalten, deren Ankunft im Zeitfenster liegt
    time_limit = shift_length * shift_count
    df_jobs_times = df_jobs_times[df_jobs_times['Arrival'] < time_limit].reset_index(drop=True)
    
    valid_jobs = set(df_jobs_times['Job'])
    df_jssp = df_jssp[df_jssp['Job'].isin(valid_jobs)].reset_index(drop=True)

    return df_jssp, df_jobs_times
    

# OLD_________________________________________________________________________________________________________________________________


def create_jobs_for_days(df_template: pd.DataFrame, day_count: int = 1, u_b_mmax: float = 0.9,
                         shuffle: bool = False, job_seed: int = 50, arrival_seed: int = 122) -> pd.DataFrame:
    """
    Erzeugt Jobs für mehrere Tage. Für jeden Tag werden 3 Jobgruppen erzeugt.

    Parameter:
    - df_template: Vorlage für Jobs mit ['Job', 'Operation', 'Machine', 'Processing Time']
    - day_count: Anzahl der Tage, für die Jobs erzeugt werden sollen
    - shuffle: Ob bei den Wiederholungen gemischt werden soll (außer jede 5.)
    - seed: Basiszufallswert (pro Wiederholung erhöht)

    Rückgabe:
    - DataFrame aller generierten Jobs mit fortlaufenden Job-IDs
    """

    # 1) Aufträge generieren
    repetitions = 4 * day_count 
    df_jssp = create_multiple_jobs(df_template, repetitions=repetitions, shuffle=shuffle, seed=job_seed)

    # 2) Mittlere Interarrival-Zeit berechnen
    t_a = gen_interarrival.calculate_mean_interarrival_time(df_jssp, u_b_mmax=u_b_mmax)

    # 3) Ankunftszeiten erzeugen (ab Tag 0)
    df_jobs_arrivals = gen_interarrival.generate_arrivals(df_jssp, mean_interarrival_time=t_a, start_time=0.0, random_seed=arrival_seed)

    # 4) Nur Jobs behalten, deren Ankunft im Zeitfenster liegt
    time_limit = 60 * 24 * day_count
    df_jobs_arrivals = df_jobs_arrivals[df_jobs_arrivals['Arrival'] < time_limit].reset_index(drop=True)
    
    valid_jobs = set(df_jobs_arrivals['Job'])
    df_jssp = df_jssp[df_jssp['Job'].isin(valid_jobs)].reset_index(drop=True)

    return df_jssp, df_jobs_arrivals

    

def create_multiple_jobs(df_template: pd.DataFrame,
                         repetitions: int = 3,
                         shuffle: bool = False,
                         seed: int = 50) -> pd.DataFrame:
    """
    Ruft `create_jobs` mehrfach auf und erzeugt eine kombinierte Menge neuer Jobs
    mit automatisch fortlaufenden Job-IDs (Job_000, Job_001, ...).

    Parameter:
    - df_template: Job-Vorlage mit ['Job', 'Operation', 'Machine', 'Processing Time']
    - repetitions: Anzahl der Wiederholungen
    - shuffle: Ob die Job-Gruppen pro Wiederholung gemischt werden sollen
    - seed: Basis-Zufallsseed (für jedes Replikat wird der Seed erhöht)

    Rückgabe:
    - Kombinierter DataFrame mit eindeutig benannten Jobs
    """
    all_jobs = []
    jobs_per_template = df_template['Job'].nunique()

    for i in range(repetitions):
        offset = i * jobs_per_template

        if (shuffle == True) and (i % 5 != 0):
            current_seed = seed + i  # z. B. 50, 51, 52 ...
            df_jobs = create_jobs(df_template, offset=offset, 
                                  shuffle=shuffle, seed=current_seed)
        else:
            
            df_jobs = create_jobs(df_template, offset=offset, shuffle=False)
            
        all_jobs.append(df_jobs)

    return pd.concat(all_jobs, ignore_index=True)




def create_jobs(df_template: pd.DataFrame,
                offset: int = 0,
                shuffle: bool = False,
                seed: int = 50) -> pd.DataFrame:
    """
    Erzeugt neue Jobs aus dem gegebenen Template mit fortlaufenden IDs.

    Parameter:
    - df_template: DataFrame mit ['Job', 'Operation', 'Machine', 'Processing Time']
    - offset: Startindex für neue Job-IDs (z.B. 0 für 'Job_000', 10 für 'Job_010')
    - shuffle: Ob die Reihenfolge der Jobgruppen gemischt werden soll
    - seed: Zufalls-Seed für das Mischen

    Rückgabe:
    - DataFrame mit neuen Jobs und eindeutigen IDs
    """

    # 1) Template-Jobs in Gruppen aufteilen
    groups = [grp for _, grp in df_template.groupby('Job', sort=False)]

    # 2) Optional mischen
    if shuffle:
        random.seed(seed)
        random.shuffle(groups)

    # 3) Neue Jobs generieren
    new_recs = []
    for i, grp in enumerate(groups):
        job_id = f"Job_{offset + i:03d}"
        for _, row in grp.iterrows():
            new_recs.append({
                'Job': job_id,
                'Operation': row['Operation'],
                'Machine': row['Machine'],
                'Processing Time': row['Processing Time']
            })

    return pd.DataFrame(new_recs).reset_index(drop=True)


    