from typing import List, Optional, Dict, Tuple

import pandas as pd
from statistics import mean

from scipy.stats import kendalltau


def get_kendall_tau_experiment_shift_df(
    df_schedules: pd.DataFrame,
    experiment_col: str = "Experiment_ID",
    shift_col: str = "Shift",
    job_col: str = "Job",
    machine_col: str = "Machine",
    start_col: str = "Start",
    comparison_start_time: Optional[float] = None,
    sort_shifts_numeric: bool = True
) -> pd.DataFrame:
    """
    Erzeugt ein DataFrame mit Kendall's Tau je (Experiment, Shift),
    wobei pro Experiment immer der aktuelle Shift x mit dem vorherigen Shift x-1 verglichen wird.

    Rückgabe-Spalten: ['Experiment_ID', 'Shift', 'Kendall_Tau']

    - comparison_start_time=None  -> kein Zeitfilter (nur gemeinsame Jobs pro Maschine)
    - comparison_start_time=t     -> filtert beide Schedules auf Start >= t, VOR dem symmetrischen Matching
    - sort_shifts_numeric=True    -> Shifts nach numerischem Wert sortieren (robust, falls als Strings gespeichert)
    """
    # Pflichtspalten prüfen
    req = {experiment_col, shift_col, job_col, machine_col, start_col}
    missing = [c for c in req if c not in df_schedules.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in df_schedules: {missing}")

    rows = []

    # Pro Experiment arbeiten
    for exp_id, df_exp in df_schedules.groupby(experiment_col, dropna=False):
        # Shifts sortieren
        unique_shifts = df_exp[shift_col].dropna().unique()
        if sort_shifts_numeric:
            # Map zu numerischen Werten, wo möglich
            tmp = [(sv, pd.to_numeric(sv, errors="coerce")) for sv in unique_shifts]
            # Nur solche mit definierter Numerik sortieren, Originalwert beibehalten
            tmp_sorted = sorted([t for t in tmp if pd.notna(t[1])], key=lambda x: x[1])
            ordered_shifts = [orig for (orig, num) in tmp_sorted]
            # Falls es auch nicht-numerische gibt, hinten in alphabetischer Ordnung anfügen
            rest = sorted([orig for (orig, num) in tmp if pd.isna(num)])
            ordered_shifts.extend([r for r in rest if r not in ordered_shifts])
        else:
            ordered_shifts = sorted(unique_shifts)

        # nacheinander x vs. x-1 vergleichen
        for i in range(1, len(ordered_shifts)):
            prev_shift = ordered_shifts[i-1]
            curr_shift = ordered_shifts[i]

            prev_sched = (
                df_exp[df_exp[shift_col] == prev_shift][[job_col, machine_col, start_col]]
                .rename(columns={job_col: "Job", machine_col: "Machine", start_col: "Start"})
            )
            curr_sched = (
                df_exp[df_exp[shift_col] == curr_shift][[job_col, machine_col, start_col]]
                .rename(columns={job_col: "Job", machine_col: "Machine", start_col: "Start"})
            )

            if prev_sched.empty or curr_sched.empty:
                tau = None
            else:
                tau = compute_mean_kendall_tau(prev_sched, curr_sched, comparison_start_time)

            rows.append({
                "Experiment_ID": exp_id,
                "Shift": curr_shift,     # aktueller Shift x (verglichen mit x-1)
                "Kendall_Tau": tau
            })

    return pd.DataFrame(rows, columns=["Experiment_ID", "Shift", "Kendall_Tau"])

def compute_mean_kendall_tau(
        previous_schedule: pd.DataFrame, new_schedule: pd.DataFrame,
        comparison_start_time: Optional[float] = None) -> Optional[float]:
    """
    Mean Kendall's Tau über alle Maschinen.
    Wenn comparison_start_time=None, wird NICHT nach Startzeit gefiltert.
    """
    machines, original_sequences, revised_sequences = _get_machines_and_sequences_dicts(
        previous_schedule, new_schedule, comparison_start_time
    )
    taus = []
    for machine in machines:
        tau = _compute_kendall_tau(original_sequences[machine], revised_sequences[machine])
        if tau is not None:
            taus.append(tau)
    return mean(taus) if taus else None


def _get_machines_and_sequences_dicts(
    previous_schedule: pd.DataFrame,
    new_schedule: pd.DataFrame,
    comparison_start_time: Optional[float] = None
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Extrahiert pro Maschine die Job-Sequenzen aus beiden Schedules.
    - Wenn comparison_start_time is not None: filtere auf Start >= Vergleichszeit.
    - Sonst: kein Zeitfilter.
    - Symmetrische Filterung auf gemeinsame Jobs je Maschine.
    """
    # Normalisieren
    df_prev = previous_schedule.rename(columns={"Start": "Start_prev", "Job": "Job"}).copy()
    df_new  = new_schedule.rename(columns={"Start": "Start_rev",  "Job": "Job"}).copy()

    # Optionaler Zeitfilter
    if comparison_start_time is not None:
        df_prev = df_prev[df_prev["Start_prev"] >= comparison_start_time]
        df_new  = df_new[df_new["Start_rev"]  >= comparison_start_time]

    # Sortierung
    df_prev_sorted = df_prev[["Machine", "Start_prev", "Job"]].sort_values(["Machine", "Start_prev"])
    df_new_sorted  = df_new[["Machine", "Start_rev",  "Job"]].sort_values(["Machine", "Start_rev"])

    # Typkonsistenz
    df_prev_sorted["Job"] = df_prev_sorted["Job"].astype(str)
    df_new_sorted["Job"]  = df_new_sorted["Job"].astype(str)

    # Sequenzen je Maschine
    prev_seqs = df_prev_sorted.groupby("Machine")["Job"].apply(list).to_dict()
    new_seqs  = df_new_sorted.groupby("Machine")["Job"].apply(list).to_dict()

    # gemeinsame Maschinen
    machines = sorted(set(prev_seqs) & set(new_seqs))

    # symmetrisch gemeinsame Jobs je Maschine
    original_sequences: Dict[str, List[str]] = {}
    revised_sequences:  Dict[str, List[str]] = {}
    for m in machines:
        orig_seq = prev_seqs[m]
        new_seq  = new_seqs[m]
        shared   = set(orig_seq) & set(new_seq)
        original_sequences[m] = [j for j in orig_seq if j in shared]
        revised_sequences[m]  = [j for j in new_seq  if j in shared]

    return machines, original_sequences, revised_sequences

def _compute_kendall_tau(original_sequence: List[str], revised_sequence: List[str]) -> Optional[float]:
    """
    Computes Kendall's Tau between two job sequences with identical elements.

    :param original_sequence: Reference job sequence.
    :param revised_sequence: Job sequence to compare (same elements, different order).
    :return: Kendall's Tau correlation coefficient rounded to 5 decimals (or 1.0 if undefined).
    """
    if len(original_sequence) != len(revised_sequence) or len(original_sequence) < 2:
        return 1.0  # Invalid input or too short for Tau

    # Map each job to its rank in the original sequence
    rank_original = {job: i for i, job in enumerate(original_sequence)}

    # Translate revised_sequence into the corresponding ranks from original_sequence
    rank_revised = [rank_original[job] for job in revised_sequence]

    # Baseline represents the ideal order (0, 1, 2, ..., n-1)
    baseline = list(range(len(rank_revised)))

    # Compare the order in revised_sequence against the original using Kendall's Tau
    tau, _ = kendalltau(baseline, rank_revised)

    return round(tau, 5) if tau is not None else 1.0