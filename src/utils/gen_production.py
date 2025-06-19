import src.utils.gen_interarrival as gen_interarrival 
import pandas as pd
import random


def create_production_orders(df_template: pd.DataFrame,
                offset: int = 0,
                shuffle: bool = False,
                seed: int = 50) -> pd.DataFrame:
    """
    Erzeugt neue Produktionsaufträge aus dem gegebenen Template mit fortlaufenden IDs.

    Jeder Produktionsauftrag basiert auf einem Arbeitsplan (Process_Plan_ID).

    Parameter:
    - df_template: DataFrame mit ['Process_Plan_ID', 'Operation', 'Machine', 'Processing Time']
    - offset: Startindex für neue Production_Order_IDs (z.B. 0 für erste Order-ID)
    - shuffle: Ob die Reihenfolge der Vorlagen-Pläne gemischt werden soll
    - seed: Zufalls-Seed für das Mischen

    Rückgabe:
    - DataFrame mit ['Production_Order_ID', 'Process_Plan_ID', 'Operation', 'Machine', 'Processing Time']
    """

    # 1) Vorlage-Pläne gruppieren
    groups = [grp for _, grp in df_template.groupby('Process_Plan_ID', sort=False)]

    # 2) Optional mischen
    if shuffle:
        random.seed(seed)
        random.shuffle(groups)

    # 3) Neue Produktionsaufträge erzeugen
    new_recs = []
    for i, grp in enumerate(groups):
        production_order_id = offset + i
        process_plan_id = grp['Process_Plan_ID'].iloc[0]
        for _, row in grp.iterrows():
            new_recs.append({
                'Production_Order_ID': production_order_id,
                'Process_Plan_ID': process_plan_id,
                'Operation': row['Operation'],
                'Machine': row['Machine'],
                'Processing Time': row['Processing Time']
            })

    return pd.DataFrame(new_recs).reset_index(drop=True)
    


def jssp_dict_to_df(jobs_dict: dict) -> pd.DataFrame:
    """
    Wandelt ein Dictionary mit Job-Operationen in einen DataFrame um.

    Parameter:
    - jobs_dict: dict
        Schlüssel sind Job-Indizes (z.B. 0, 1, 2),
        Werte sind Listen von [Machine, Processing Time].

    Rückgabe:
    - pd.DataFrame mit Spalten ['Process_Plan_ID', 'Operation', 'Machine', 'Processing Time'].
      Die Spalte 'Operation' enthält die Reihenfolge der Operation innerhalb des Plans.
    """
    records = []
    for plan_id, ops in jobs_dict.items():
        for op_idx, (machine_idx, proc_time) in enumerate(ops):
            records.append({
                'Process_Plan_ID': plan_id,
                'Operation': op_idx,
                'Machine': f'M{machine_idx:02d}',
                'Processing Time': proc_time
            })
    df = pd.DataFrame(records, columns=['Process_Plan_ID', 'Operation', 'Machine', 'Processing Time'])
    return df