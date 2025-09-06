import math
import random
import re
import seaborn as sns
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Literal
from cycler import cycler
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch

from src.domain.orm_models import Job


class DataPreprocessor:
    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated.")

    @staticmethod
    def _step1_exclude_initial_text(content: str, skip_until_marker: int = 1) -> str:
        """
        Removes the text up to and including the N-th line that contains multiple '+' characters.

        :param content: The full text content.
        :param skip_until_marker: Index of the '+++' line after which the text should be kept.
        :return: The remaining text starting after the specified marker line.
        """
        # Find all lines containing +++
        matches = list(re.finditer(r"\n.*\+{3,}.*\n", content))

        # Keep everything after the N-th +++ line
        return content[matches[skip_until_marker].end():]

    @staticmethod
    def _step2_parse_text_with_instances_to_dict(content: str, verbose: bool = False) -> dict:
        """
        Parses a structured text with alternating instance names and data blocks into a dictionary.

        :param content: A string containing instance descriptions and matrix blocks separated by '+++' lines.
        :param verbose: If True, enables debug output (optional).
        :return: A dictionary where keys are instance descriptions
            and values are the corresponding matrix blocks (as strings).
        """

        # Separate blocks using +++ lines and remove unnecessary spaces
        raw_blocks = [block.strip() for block in re.split(r"\n.*\+{3,}.*\n", content) if block.strip()]

        if verbose:
            print("====== Raw blocks example ======")
            for i, b in enumerate(raw_blocks[:4]):
                print(f"--- {b} ---\n") if i % 2 == 0 else print(b, "\n")
            print("=" * 20)

        # Ensure that the number of blocks is even
        if len(raw_blocks) % 2 != 0:
            raise ValueError("Number of blocks is odd – each instance requires exactly 2 blocks (description + matrix)")

        # Build dictionary
        instance_dict = {}

        for i in range(0, len(raw_blocks), 2):
            key = raw_blocks[i].strip()  # e.g. "instance abz5"
            lines = raw_blocks[i + 1].splitlines()  # contains matrix block including matrix-info
            cleaned_lines = lines[2:]  # remove matrix-info (e.g. 10 10)
            matrix_block = "\n".join(cleaned_lines)  # reassemble the matrix
            instance_dict[key] = matrix_block

        return instance_dict


    @staticmethod
    def _step3_structure_dict(raw_dict: Dict[str, str]) -> Dict[str, Dict[int, List[Dict[str, int]]]]:
        """
        :param raw_dict: Dictionary mapping instance names to whitespace-separated job routing strings.
        :return: Nested dictionary with structured operation dictionaries.
        """
        structured_dict = {}
        for instance_name, matrix_text in raw_dict.items():
            lines = matrix_text.strip().splitlines()
            jobs = {}
            for job_id, line in enumerate(lines):
                try:
                    numbers = list(map(int, line.strip().split()))
                    job_ops = [{"machine": numbers[i], "duration": numbers[i + 1]} for i in range(0, len(numbers), 2)]
                    jobs[job_id] = job_ops
                except ValueError:
                    continue
            structured_dict[instance_name] = jobs
        return structured_dict

    @classmethod
    def transform_file_to_instances_dictionary(cls, file_path: Path) -> dict:
        # Read file
        file = open(file_path, encoding="utf-8")
        content = file.read()
        file.close()

        content_without_introduction = cls._step1_exclude_initial_text(content)

        # Dictionary with instances as keys and matrix as value (string)
        instances_string_dict = cls._step2_parse_text_with_instances_to_dict(content_without_introduction)

        # Dictionary with instances as keys and matrix as value (dictionary/JSON of routings)
        return cls._step3_structure_dict(instances_string_dict)


    @staticmethod
    def routing_dict_to_df(
            routings_dict: dict, routing_column: str = 'Routing_ID', operation_column: str = 'Operation',
            machine_column: str = "Machine", duration_column: str = "Processing Time",) -> pd.DataFrame:
        """
        Converts a routing dictionary with structured operations into a pandas DataFrame.

        :param routings_dict: Dictionary where each key is a routing ID (e.g., 0, 1, 2)
                              and each value is a list of operations as {"machine": int, "duration": int}.
        :param routing_column: Name of the column that will store the routing ID.
        :param operation_column: Name of the column that will store the operation index.
        :param machine_column: Name of the column that will store the machine name (e.g., ``'M00'``).
        :param duration_column: Name of the column that will store the processing time.

        :return: DataFrame with one row per operation, including routing ID, operation index, machine, and processing time.
        """
        records = []
        for plan_id, ops in routings_dict.items():
            for op_idx, op in enumerate(ops):
                records.append({
                    routing_column: plan_id,
                    operation_column: op_idx,
                    machine_column: f'M{op["machine"]:02d}',
                    duration_column: op["duration"]
                })
        df = pd.DataFrame(records, columns=[routing_column, operation_column, machine_column, duration_column])
        return df


class SimulationDataVisualization:

    @staticmethod
    def duration_log_normal(duration: float, sigma: float = 0.2, round_decimal: int = 0,
                            seed: Optional[int] = None) -> float:
        """
        Returns a log-normal distributed duration whose expected value in real space
        is approximately equal to the given `duration`.

        :param duration: Expected mean duration in real space.
        :param sigma: Standard deviation in log space (must be non-negative).
        :param round_decimal: Number of decimal places to round the result.
        :param seed: Optional random seed for reproducibility.
        :return: A sampled duration from the log-normal distribution.
        """
        sigma = max(sigma, - sigma)

        rnd_numb_gen = random.Random(seed) if seed is not None else random

        # Compute mu so that the expected value in real space is approximately equal to `duration` (E[X] ≈ duration)
        mu = math.log(duration) - 0.5 * sigma ** 2

        result = rnd_numb_gen.lognormvariate(mu, sigma)

        return round(result, round_decimal)

    @staticmethod
    def _seed_from_row(job_id: str, operation: int) -> int:
        """
        Builds a deterministic seed from (Job, Operation).

        Expected job format: <value1>-<value2>-<value3>.
        Uses only <value1> and <value3>.
        """
        job_id = str(job_id)
        # Extract first and last numeric group
        parts = re.findall(r"\d+", job_id)
        if len(parts) >= 2:
            job_digits = f"{parts[0]}{parts[-1]}"
        else:
            job_digits = "".join(parts)  # fallback

        return int(f"{job_digits}{int(operation):02d}")  # pad op to 2 digits


    @staticmethod
    def make_jobs_dataframe(jobs: Iterable[Job]) -> pd.DataFrame:
        """
        Build a DataFrame with job operations and simulated durations.
        """
        records = []
        for job in jobs:
            for op in job.operations:  # each operation individually
                records.append({
                    "Job": job.id,
                    "Routing_ID": job.routing_id,
                    "Operation": op.position_number,
                    "Duration": op.duration
                #    "Simulation Duration": simulated_duration_from_op(op, sigma=sigma, round_decimal=round_decimal)
                })
        df_jobs = pd.DataFrame(records).sort_values(by=["Job", "Operation"]).reset_index(drop=True)
        return df_jobs

    @classmethod
    def make_unique_jobs_dataframe(cls, jobs: Iterable[Job]) -> pd.DataFrame:
        """
        Build a DataFrame with job operations
        """
        df_jobs = cls.make_jobs_dataframe(jobs)

        # Regex: drei Gruppen mit Ziffern, Bindestriche dazwischen
        df_jobs["Job"] = df_jobs["Job"].str.replace(
            r"^(\d+)-\d+-(\d+)$", r"\1-\2", regex=True
        )

        # Keine Duplikate
        df_jobs = df_jobs.drop_duplicates(subset=["Job"], keep="first").reset_index(drop=True)
        return df_jobs

    # --- Style-Setter ---
    @staticmethod
    def set_latex_style(mono: bool = False, axis_grid: Literal["both", "x", "y"] = "both", spines:bool = True):
        """
        Set LaTeX-like clean style.
        - Horizontal grid on y-axis
        - Native ticks
        - Colors (Tab10) or monochrome gray tones if mono=True
        """
        sns.set_style("white")
        sns.set_context("paper")
        rcParams.update({
            "axes.grid": True,  # horizontale Gitterlinien
            "axes.grid.axis": axis_grid,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,

            "axes.spines.top": spines,  # obere Rahmenlinie weg/da
            "axes.spines.right": spines,  # rechte Rahmenlinie weg/da

            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "0.85",

            "lines.linewidth": 1.6,

            # Ticks: unten/links sichtbar, innen gerichtet
            "xtick.bottom": True,
            "xtick.top": False,
            "ytick.left": False,
            "ytick.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
        })

        if mono:
            rcParams["axes.prop_cycle"] = cycler(color=["0.15", "0.35", "0.55", "0.75"])

    @classmethod
    def _plot_deviation_kde(
            cls, df_jobs, sigmas, round_decimal: int = 0,
            x_min: Optional[float] = None, x_max: Optional[float] = None, x_step: Optional[float] = None,
            job_column: str = "Job", op_column: str = "Operation", y_max: Optional[float] = None,
            mode: Literal["relative", "absolute"] = "relative", x_font_size: int = 10, with_arrows:bool = False):
        """
        Plot KDEs of deviations for different sigma values.
        - If mode="relative": (sim - orig) / orig
        - If mode="absolute": (sim - orig)
        Uses deterministic seeds per (Job, Operation).
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        palette = get_gray_palette()

        for sigma, color in zip(sigmas, palette):
            diffs = []
            for _, row in df_jobs.iterrows():
                d = row["Duration"]
                seed = cls._seed_from_row(row[job_column], row[op_column])
                sim = cls.duration_log_normal(d, sigma=sigma, round_decimal=round_decimal, seed=seed)

                if mode == "relative":
                    diffs.append((sim - d) / d)
                else:  # absolute
                    diffs.append(sim - d)

            sns.kdeplot(diffs, linewidth=2, cut=0, ax=ax, label=fr"$\sigma={sigma}$", color=color)
            color = ax.get_lines()[-1].get_color()

            # Min/Max in gleicher Farbe
            xmin, xmax = min(diffs), max(diffs)
            ax.axvline(xmin, color=color, linestyle="--", alpha=0.8, linewidth=1)
            ax.axvline(xmax, color=color, linestyle="--", alpha=0.8, linewidth=1)

        # Labels abhängig vom Modus
        if mode == "relative":
            xlabel = r"$\frac{\mathrm{simulierte\ Dauer}\ -\ \mathrm{originale\ Dauer}}{\mathrm{originale\ Dauer}}$"
            ax.set_ylabel(r"$\mathrm{Dichte}$", fontsize=10)  # keine Einheit
        else:
            xlabel = r"$\mathrm{simulierte\ Dauer}\ -\ \mathrm{originale\ Dauer}\ [\mathrm{min}]$"
            ax.set_ylabel(r"$\mathrm{Dichte\ [1/min]}$", fontsize=10)

        ax.set_xlabel(xlabel, fontsize=x_font_size)

        ax.legend(bbox_to_anchor=(0.898, 0.99), loc="upper left")

        # Optionale X-Achsenbegrenzung und Ticks
        if x_min is not None or x_max is not None:
            ax.set_xlim(left=x_min, right=x_max)
        if x_step is not None:
            left, right = ax.get_xlim()
            left = left if x_min is None else x_min
            right = right if x_max is None else x_max
            ax.set_xticks(np.arange(left, right + 0.001, x_step))

        # Spines sichtbar
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_linewidth(1)
        ax.spines["left"].set_linewidth(1)


        if y_max:
            ax.set_ylim(top=y_max)

        # Spines anpassen: linestyle und alpha
        for side in ["top", "right"]:
            ax.spines[side].set_linestyle("--")
            ax.spines[side].set_alpha(0.14)
            ax.spines[side].set_linewidth(0.8)


        if with_arrows:
            # Pfeilspitzen an Achsen
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            arrow_x = FancyArrowPatch(
                (xmax, ymin), (xmax + 0.01 * (xmax - xmin), ymin),
                arrowstyle="-|>", mutation_scale=14,
                lw=0, color="black", clip_on=False
            )
            ax.add_patch(arrow_x)

            arrow_y = FancyArrowPatch(
                (xmin, ymax), (xmin, ymax + 0.01 * (ymax - ymin)),
                arrowstyle="-|>", mutation_scale=14,
                lw=0, color="black", clip_on=False
            )
            ax.add_patch(arrow_y)

        return fig

    # Wrapper für relative Abweichung
    @classmethod
    def plot_relative_deviation_kde(
            cls,
            df_jobs,
            sigmas,
            round_decimal: int = 0, y_max: Optional[float] = None,
            x_min: Optional[float] = None,
            x_max: Optional[float] = None,
            x_step: Optional[float] = None,
            job_column: str = "Job",
            op_column: str = "Operation",
            x_font_size: int = 10, with_arrows:bool = False
    ):
        """
        Plot KDEs der relativen Abweichung:
        (simulierte Dauer − originale Dauer) / originale Dauer
        """
        return cls._plot_deviation_kde(
            df_jobs, sigmas,
            round_decimal=round_decimal, y_max = y_max,
            x_min=x_min, x_max=x_max, x_step=x_step,
            job_column=job_column, op_column=op_column,
            mode="relative", x_font_size = x_font_size, with_arrows = with_arrows
        )

    # Wrapper für absolute Abweichung
    @classmethod
    def plot_absolute_deviation_kde(
            cls,
            df_jobs,
            sigmas,
            round_decimal: int = 0, y_max: Optional[float] = None,
            x_min: Optional[float] = None,
            x_max: Optional[float] = None,
            x_step: Optional[float] = None,
            job_column: str = "Job",
            op_column: str = "Operation",
            x_font_size: int = 10, with_arrows:bool = False
    ):
        """
        Plot KDEs der absoluten Abweichung:
        simulierte Dauer − originale Dauer
        """
        return cls._plot_deviation_kde(
            df_jobs, sigmas,
            round_decimal=round_decimal,
            x_min=x_min, x_max=x_max, x_step=x_step,
            job_column=job_column, op_column=op_column,
            mode="absolute", y_max = y_max,
            x_font_size=x_font_size, with_arrows = with_arrows
        )


    @classmethod
    def add_simulated_durations(
            cls,
            df_jobs: pd.DataFrame,
            sigmas: list[float],
            round_decimal: int = 0,
            job_column: str = "Job",
            op_column: str = "Operation"
    ) -> pd.DataFrame:
        """
        Fügt dem DataFrame zusätzliche Spalten mit simulierten Dauern
        für verschiedene Sigma-Werte hinzu.

        Beispiel: 'Duration_sigma0.1', 'Duration_sigma0.2', ...
        """
        df = df_jobs.copy()

        for sigma in sigmas:
            col_name = f"Duration_sigma{sigma}"
            values = []

            for _, row in df.iterrows():
                d = row["Duration"]
                seed = cls._seed_from_row(row[job_column], row[op_column])
                sim = cls.duration_log_normal(d, sigma=sigma, round_decimal=round_decimal, seed=seed)
                values.append(sim)

            df[col_name] = values

        return df

    @classmethod
    def summarize_duration_differences(cls, df_jobs: pd.DataFrame, sigmas: list[float],
                                       with_negative: bool = False) -> pd.DataFrame:
        """
        Summarize percentage of absolute differences (Duration_sigmaX - Duration)
        """
        df = cls.add_simulated_durations(df_jobs, sigmas=sigmas)

        # Bins & Labels (Absolute Abweichung)
        bins = [-float("inf"), 0, 5, 10, 30, 60, float("inf")]
        labels = [
            r"$<0\,\text{min}$",
            r"$0\!-\!5\,\text{min}$",
            r"$5\!-\!10\,\text{min}$",
            r"$10\!-\!30\,\text{min}$",
            r"$30\!-\!60\,\text{min}$",
            r"$>60\,\text{min}$"
        ]

        result = {}

        for col in df.columns:
            if col.startswith("Duration_sigma"):
                diffs = df[col] - df["Duration"]
                cats = pd.cut(diffs, bins=bins, labels=labels, right=True)
                counts = cats.value_counts().reindex(labels, fill_value=0)
                percentages = (counts / counts.sum() * 100).round(2)

                # "<0" entfernen
                if with_negative is False:
                    percentages = percentages.drop("<0")

                result[col] = percentages
        summary = pd.DataFrame(result)
        summary.index.name = "Absolute Abweichung [min]"
        return summary

    @classmethod
    def plot_deviation_sigma_summary(cls, df_jobs: pd.DataFrame, sigmas: list[float], with_negative: bool = False,
                                     bar_width: float = 0.22,
                                     gap: float = 0.1,
                                     label_threshold: float = 0.01,
                                     y_max:int  = 100):
        """
        Gruppierte Balken (in %) je Sigma.
        df_summary: Index = Klassen ("<0","0–5","5–10","10–30","30–60",">60"),
                    Columns = "Duration_sigmaX", Werte = Prozentanteile.
        - "<0" wird entfernt
        - x-Achse zeigt nur den Sigma-Wert (0.1, 0.2, …)
        - slot = bar_width * n_classes + gap
        """

        data = cls.summarize_duration_differences(df_jobs, sigmas=sigmas, with_negative = with_negative)

        # Klassen vorbereiten
        classes = data.index
        # data = data.loc[classes]

        # Sigmas numerisch sortieren
        cols = list(data.columns)
        sig_vals = [float(c.split("sigma")[-1]) for c in cols]
        order = np.argsort(sig_vals)
        cols = [cols[i] for i in order]
        sig_vals = [sig_vals[i] for i in order]
        data = data[cols]

        # Positionen berechnen
        n_sig = len(cols)
        n_cls = len(classes)
        slot = bar_width * n_cls + gap
        x_centers = np.arange(n_sig, dtype=float) * slot

        # Offsets für jede Klasse innerhalb des Slots
        offsets = (np.arange(n_cls) - (n_cls - 1) / 2) * bar_width

        fig, ax = plt.subplots(figsize=(12, 6))
        palette = sns.color_palette("muted", n_colors=n_cls)



        for k, (value_cls, color) in enumerate(zip(classes, palette)):
            vals = data.loc[value_cls].values.astype(float)
            x = x_centers + offsets[k]
            bars = ax.bar(x, vals, width=bar_width, color=color, label=value_cls)

            # Prozentlabels in die Balken
            for rect, v in zip(bars, vals):
                if v >= label_threshold:
                    y = rect.get_y() + rect.get_height() / 2 if v >= 1.4 else rect.get_y() + rect.get_height() + 0.4
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        y,
                        f"{v:.1f}%",
                        ha="center", va="center", fontsize=9, color="black", clip_on=False
                    )

        # Achsen & Rahmen
        ax.set_ylim(0, y_max)
        ax.set_xticks(x_centers)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_xticklabels([f"{s:g}" for s in sig_vals], fontsize=9)
        ax.set_xlabel(r"$\sigma$", fontsize=10)
        ax.set_ylabel("Anteil [%]", fontsize=10)
        #ax.legend(title="Abweichung [min]", loc="upper right")
        ax.legend(bbox_to_anchor=(0.898, 0.99), loc="upper left")

        # X-Limits passend zu Slotbreite
        ax.set_xlim(x_centers[0] - slot / 2 + gap / 2, x_centers[-1] + slot / 2 - gap / 2)


        ax.spines["bottom"].set_linewidth(1)
        ax.spines["left"].set_linewidth(1)

        for side in ["top", "right"]:
            sp = ax.spines[side]
            sp.set_visible(False)

        fig.tight_layout()
        return fig



def get_gray_palette():
    return [
    "#4c6a91",  # graublau
    "#915c5c",  # graurot
    "#5c7a5c",  # graugrün
    "#6d5c91",  # grauviolett
    "#91885c"   # graugelb
]
