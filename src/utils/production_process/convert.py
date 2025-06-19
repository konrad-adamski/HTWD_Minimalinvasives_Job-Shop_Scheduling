import pandas as pd

def jssp_dict_to_df(jobs_dict: dict) -> pd.DataFrame:
    """
    Wandelt ein Dictionary mit Job-Operationen in einen DataFrame um.

    Parameter:
    - jobs_dict: dict
        Schlüssel sind Job-Indizes (z.B. 0, 1, 2),
        Werte sind Listen von [Machine, Processing Time].

    Rückgabe:
    - pd.DataFrame mit Spalten ['Production_Plan_ID', 'Operation', 'Machine', 'Processing Time'].
      Die Spalte 'Operation' enthält die Reihenfolge der Operation innerhalb des Plans.
    """
    records = []
    for plan_id, ops in jobs_dict.items():
        for op_idx, (machine_idx, proc_time) in enumerate(ops):
            records.append({
                'Production_Plan_ID': plan_id,
                'Operation': op_idx,
                'Machine': f'M{machine_idx:02d}',
                'Processing Time': proc_time
            })
    df = pd.DataFrame(records, columns=['Production_Plan_ID', 'Operation', 'Machine', 'Processing Time'])
    return df