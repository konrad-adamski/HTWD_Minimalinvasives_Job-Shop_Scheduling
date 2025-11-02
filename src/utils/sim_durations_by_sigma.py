import seaborn as sns
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from typing import Optional, Iterable, Literal
from cycler import cycler
from matplotlib import rcParams

from src.domain.orm_models import Job
from src.simulation.LognormalFactorGenerator import LognormalFactorGenerator


class SimulationDataVisualization:

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
            job_column:str = "Job", operation_column: str = "Operation",
            duration_column: str = "Duration", y_max: Optional[float] = None,
            mode: Literal["relative", "absolute"] = "relative", x_font_size: int = 10, seed: int = 42):
        """
        Plot KDEs of deviations for different sigma values.
        - If mode="relative": (sim - orig) / orig
        - If mode="absolute": (sim - orig)
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        palette = get_gray_palette()

        df_jobs = df_jobs.sort_values(by=[job_column, operation_column]).reset_index(drop=True)
        for sigma, color in zip(sigmas, palette):

            lognormal_factor_gen = LognormalFactorGenerator(sigma=sigma, seed=seed)
            factors = lognormal_factor_gen.sample_many(len(df_jobs))
            df_jobs[f"Simulation {duration_column}"] = (df_jobs[duration_column] * pd.Series(factors)).round(0).astype(int)

            diffs = []
            for _, row in df_jobs.iterrows():
                routing_duration = row[duration_column]
                sim_duration = row[f"Simulation {duration_column}"]

                if mode == "relative":
                    diffs.append((sim_duration - routing_duration) / routing_duration)
                else:  # absolute
                    diffs.append(sim_duration - routing_duration)

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
            operation_column: str = "Operation",
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
            job_column=job_column, operation_column=operation_column,
            mode="relative", x_font_size = x_font_size
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
            operation_column: str = "Operation",
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
            job_column=job_column, operation_column=operation_column,
            mode="absolute", y_max = y_max,
            x_font_size=x_font_size
        )


    @classmethod
    def add_simulated_durations(
            cls, df_jobs: pd.DataFrame,sigmas: list[float], job_column: str = "Job",
            operation_column: str = "Operation", duration_column: str = "Duration", seed: int = 42) -> pd.DataFrame:
        """
        Fügt dem DataFrame zusätzliche Spalten mit simulierten Dauern
        für verschiedene Sigma-Werte hinzu.

        Beispiel: 'Duration_sigma0.1', 'Duration_sigma0.2', ...
        """
        df = df_jobs.sort_values(by=[job_column, operation_column]).reset_index(drop=True)

        for sigma in sigmas:
            sim_dur_col = f"Duration_sigma{sigma}"
            log_normal_factor_gen = LognormalFactorGenerator(sigma=sigma, seed=seed)
            factors = log_normal_factor_gen.sample_many(len(df))
            df[sim_dur_col] = (df[duration_column] * pd.Series(factors)).round(0).astype(int)

        return df


    @classmethod
    def summarize_duration_differences(
            cls, df_jobs: pd.DataFrame,
            sigmas: list[float], with_negative: bool = False) -> pd.DataFrame:
        """
        Summarize percentage of absolute differences (Duration_sigmaX - Duration)
        """
        df = cls.add_simulated_durations(df_jobs, sigmas=sigmas)

        # Bins & Labels (Absolute Abweichung)
        bins = [-float("inf"), -45, -30, -15, 0, 15, 30, 45, float("inf")]
        labels = [
            r"$<-45\,\text{min}$",
            r"$-45\!-\!-30\,\text{min}$",
            r"$-30\!-\!-15\,\text{min}$",
            r"$-15\!-\!0\,\text{min}$",
            r"$0\!-\!15\,\text{min}$",
            r"$15\!-\!30\,\text{min}$",
            r"$30\!-\!45\,\text{min}$",
            r"$>45\,\text{min}$"
        ]

        result = {}
        for col in df.columns:
            if col.startswith("Duration_sigma"):
                diffs = df[col] - df["Duration"]
                cats = pd.cut(diffs, bins=bins, labels=labels, right=False)   # False: [-45, -30); True: (-45, -30]
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
                                     gap: float = 0.2,
                                     label_threshold: float = 0.00,
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
