


from typing import Optional, Tuple, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize

ExtendType = Literal["auto", "both", "min", "max", None]

from fractions import Fraction

def ratio_label(value: float, max_denominator: int = 10) -> str:
    frac = Fraction(value).limit_denominator(max_denominator)
    t = frac.numerator
    e = frac.denominator - t
    return f"{t}:{e}"


def plot_experiment_heatmaps(
        df: pd.DataFrame,
        *,
        value_col: str,
        x_col: str,
        y_col: str,
        col_col: Optional[str] = None,
        row_col: Optional[str] = None,
        # Anzeigenamen (Labels) – OPTIONAL:
        value_as: Optional[str] = None,
        x_col_as: Optional[str] = None,
        y_col_as: Optional[str] = None,
        col_col_as: Optional[str] = None,
        row_col_as: Optional[str] = None,
        # Darstellung:
        cmap_name: str = "RdYlGn",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        annot: bool = False,
        fmt: str = ".2f",
        text_color: str = "black",
        figsize_scale: Tuple[float, float] = (4.8, 4.2),
        legend_steps: int = 6,
        higher_is_better: bool = True,
        auto_reverse_cmap: bool = True,
        extend: ExtendType = "auto",
        colorbar_fraction: float = 0.04,
        colorbar_pad: float = 0.02,
        title: Optional[str] = None,
        fontsize: int = 13,
):
    # 0) Effektive Labels bestimmen
    value_label = value_as or value_col
    x_label = x_col_as or x_col
    y_label = y_col_as or y_col
    col_label = col_col_as or (col_col if col_col is not None else "")
    row_label = row_col_as or (row_col if row_col is not None else "")

    # 1) Facetten-Keys (optional)
    unique_cols = [None] if col_col is None else sorted(df[col_col].unique())
    unique_rows = [None] if row_col is None else sorted(df[row_col].unique())
    n_cols, n_rows = len(unique_cols), len(unique_rows)

    # 2) Wertebereich
    z_all = df[value_col].to_numpy(dtype=float)
    data_min, data_max = float(np.nanmin(z_all)), float(np.nanmax(z_all))
    if vmin is None: vmin = data_min
    if vmax is None: vmax = data_max
    if vmin > vmax:
        raise ValueError("vmin darf nicht größer als vmax sein.")

    # 3) Colormap + ggf. Richtungsumkehr
    base_cmap = mpl.colormaps.get_cmap(cmap_name)
    cmap = base_cmap
    if auto_reverse_cmap:
        ends_with_r = cmap_name.endswith("_r")
        want_reversed = not higher_is_better
        if want_reversed ^ ends_with_r:
            cmap = base_cmap.reversed()
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    # 4) Diskrete Stufen
    x_levels = list(np.sort(df[x_col].unique()))
    y_levels = list(np.sort(df[y_col].unique()))

    # 5) Figure/Axes
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_scale[0] * n_cols, figsize_scale[1] * n_rows),
        constrained_layout=True, sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)

    for i, r in enumerate(unique_rows):
        for j, c in enumerate(unique_cols):
            ax = axes[i, j]
            sub = df
            if row_col is not None:
                sub = sub[sub[row_col] == r]
            if col_col is not None:
                sub = sub[sub[col_col] == c]

            if sub.empty:
                ax.set_visible(False)
                continue

            pivot = (
                sub.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc="mean")
                .reindex(index=y_levels, columns=x_levels)
            )
            ny, nx = pivot.shape

            X, Y = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1))
            ax.pcolormesh(
                X, Y, pivot.values,
                cmap=cmap, norm=norm,
                shading="flat", edgecolors="none"
            )

            ax.set_xticks(np.arange(nx) + 0.5)
            ax.set_yticks(np.arange(ny) + 0.5)
            ax.set_xticklabels([ratio_label(x) for x in x_levels])
            ax.set_yticklabels([ratio_label(y) for y in y_levels])


            # Spaltentitel (falls vorhanden)
            if i == 0 and col_col is not None:
                #ax.set_title(f"{col_label} = {c}")
                ax.set_title(
                    f"{col_label} = {c}",
                    fontsize=12,
                    pad = 10,
                    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
                )


            # Y-Label links
            if j == 0:
                if row_col is not None:
                    #ax.set_ylabel(f"{row_label} = {r}\n{y_label}")
                    # 1️⃣ Eingeboxtes Label oberhalb
                    ax.text(
                        -0.25, 0.5,                    # Position in Achsenkoordinaten (x<0 = links außerhalb)
                        f"{row_label} = {r}",          # nur der obere Teil
                        transform=ax.transAxes,        # in Achsenkoordinaten platzieren
                        fontsize=12,
                        ha="center", va="center",
                        rotation=90,                   # damit es wie ein Achsenlabel wirkt
                        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
                    )

                    # 2️⃣ Normales y-Label unten drunter
                    ax.set_ylabel(
                        y_label,
                        fontsize=12,
                        labelpad=15
                    )

                else:
                    ax.set_ylabel(y_label)

            ax.set_xlabel(x_label)

            if annot:
                vals = pivot.values
                for yi in range(ny):
                    for xi in range(nx):
                        v = vals[yi, xi]
                        if np.isfinite(v):
                            ax.text(xi + 0.5, yi + 0.5, format(v, fmt),
                                    ha="center", va="center", color=text_color, fontsize=fontsize)

    # 6) Colorbar + extend (ohne „Spiegeln“)
    if extend == "auto":
        _extend = None
        if data_min < vmin and data_max > vmax:
            _extend = "both"
        elif data_min < vmin:
            _extend = "min"
        elif data_max > vmax:
            _extend = "max"
    else:
        _extend = extend

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        orientation="vertical",
        fraction=colorbar_fraction,
        pad=colorbar_pad,
        label=value_label,
        extend=_extend,
    )

    ticks = np.linspace(vmin, vmax, legend_steps)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t:.2f}" for t in ticks])

    # Optionaler Supertitel
    if title is not None:
        fig.suptitle(title, fontsize=fontsize)

    return fig, axes


# Wrapper: Kendall Tau (hoch = gut)
def plot_experiment_heatmaps_kendall_tau(
        df: pd.DataFrame,
        *,
        value_col: str, value_as: str ="Kendall τ",
        x_col: str = "Inner Tardiness Ratio", x_col_as: Optional[str] = None,
        y_col: str = "Abs Lateness Ratio", y_col_as: Optional[str] = None,  # <- korrigiert
        col_col: Optional[str] = None, col_col_as: Optional[str] = None,
        row_col: Optional[str] = None, row_col_as: Optional[str] = None,
        vmin: Optional[float] = 0.7,
        vmax: Optional[float] = 1.0,
        fmt: str = ".2f",
        extend: ExtendType = "both",
        title: Optional[str] = None,
        fontsize: int = 13,
        legend_steps: int = 6,
):
    return plot_experiment_heatmaps(
        df=df,
        value_col=value_col, value_as=value_as,
        x_col=x_col, y_col=y_col,
        col_col=col_col, row_col=row_col,
        x_col_as=x_col_as, y_col_as=y_col_as,
        col_col_as=col_col_as, row_col_as=row_col_as,
        vmin=vmin, vmax=vmax,
        fmt=fmt, extend=extend,
        annot=True,
        cmap_name="RdYlGn",
        higher_is_better=True,
        auto_reverse_cmap=True,
        title=title,
        fontsize = fontsize,
        legend_steps = legend_steps
    )


# Wrapper: niedrig = gut (z. B. Tardiness, Loss, Error)
def plot_experiment_heatmaps_good_low(
        df: pd.DataFrame,
        *,
        value_col: str, value_as: Optional[str] = None,
        x_col: str,
        y_col: str,
        col_col: Optional[str] = None, col_col_as: Optional[str] = None,
        row_col: Optional[str] = None, row_col_as: Optional[str] = None,
        x_col_as: Optional[str] = None,
        y_col_as: Optional[str] = None,
        cmap_name: str = "RdYlGn",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        fmt: str = ".2f",
        extend: ExtendType = "auto",
        title: Optional[str] = None,
        **kwargs
):
    return plot_experiment_heatmaps(
        df=df,
        value_col=value_col,
        x_col=x_col, y_col=y_col,
        col_col=col_col, row_col=row_col,
        value_as=value_as,
        x_col_as=x_col_as, y_col_as=y_col_as,
        col_col_as=col_col_as, row_col_as=row_col_as,
        vmin=vmin, vmax=vmax,
        fmt=fmt, extend=extend,
        cmap_name=cmap_name,
        higher_is_better=False,
        auto_reverse_cmap=True,
        title=title,
        **kwargs
    )


def plot_experiment_boxrow(
    df_values: pd.DataFrame,
    df_meta: pd.DataFrame,
    *,
    value_col: str,
    id_col: str = "Experiment_ID",
    x_col: str = "Inner Tardiness Ratio",
    col_col: Optional[str] = None,
    value_as: Optional[str] = None,
    x_col_as: Optional[str] = None,
    col_col_as: Optional[str] = None,
    figsize_scale: Tuple[float, float] = (5.5, 4.0),
    show_median_labels: bool = True,
    median_fmt: str = ".2f",
    flier_visible: bool = True,
    fontsize: int = 12,
    title: Optional[str] = None,
    median_text_color: str = "#d95f02",
    # NEU: In welcher Spalte das x-Achsenlabel stehen soll (0-basiert)
    xlabel_at_col: int = 0,
):
    value_label = value_as or value_col
    x_label = x_col_as or x_col
    col_label = col_col_as or (col_col if col_col else "")

    meta_cols = [c for c in [id_col, x_col, col_col] if c is not None]
    if id_col not in df_values.columns:
        raise ValueError(f"`df_values` braucht die Spalte `{id_col}` für den Merge.")
    if any(c not in df_meta.columns for c in meta_cols):
        missing = [c for c in meta_cols if c not in df_meta.columns]
        raise ValueError(f"`df_meta` fehlt/fehlen: {missing}")

    dfm = df_values.merge(df_meta[meta_cols], on=id_col, how="left")

    x_levels = list(np.sort(dfm[x_col].dropna().unique()))
    unique_cols = [None] if col_col is None else sorted(dfm[col_col].dropna().unique())
    n_cols = len(unique_cols)

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(figsize_scale[0] * n_cols, figsize_scale[1]),
        constrained_layout=True, sharex=False, sharey=True
    )
    axes = np.atleast_1d(axes)

    # Stildefinitionen
    boxprops = dict(linewidth=1.2, color="black")
    whiskerprops = dict(linewidth=1.1, color="black")
    capprops = dict(linewidth=1.1, color="black")
    medianprops = dict(linewidth=1.5, color=median_text_color)
    flierprops = dict(
        marker='o', markersize=6, markerfacecolor='none',
        markeredgecolor='gray', alpha=0.6
    )

    # Sicherheitscheck für xlabel_at_col
    if n_cols > 1:
        xlabel_at_col = int(np.clip(xlabel_at_col, 0, n_cols - 1))
    else:
        xlabel_at_col = 0

    for j, c in enumerate(unique_cols):
        ax = axes[j]
        sub = dfm
        if col_col is not None:
            sub = sub[sub[col_col] == c]

        data_arrays = [
            sub.loc[sub[x_col] == xv, value_col].dropna().to_numpy()
            for xv in x_levels
        ]
        positions = np.arange(1, len(x_levels) + 1)

        bp = ax.boxplot(
            data_arrays,
            positions=positions,
            widths=0.7,
            showfliers=flier_visible,
            patch_artist=False,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops,
            flierprops=flierprops,
        )

        # ✅ Ticks + Ticklabels bei allen
        ax.set_xticks(positions)
        ax.set_xticklabels([ratio_label(x) for x in x_levels], fontsize=fontsize - 1)

        # ❌ x-Achsenlabel nur in gewünschter Spalte
        if j == xlabel_at_col:
            ax.set_xlabel(x_label, fontsize=fontsize)
        else:
            ax.set_xlabel("")

        # Y-Achse links
        if j == 0:
            ax.set_ylabel(value_label, fontsize=fontsize)

        # Spaltentitel
        if col_col is not None:
            ax.set_title(
                f"{col_label} = {c}",
                fontsize=fontsize,
                bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
                pad=8
            )

        # Mediane beschriften
        if show_median_labels:
            for med_line in bp["medians"]:
                xm = float(np.mean(med_line.get_xdata()))
                ym = float(np.mean(med_line.get_ydata()))
                ax.text(
                    xm, ym, format(ym, median_fmt),
                    ha="center", va="bottom",
                    fontsize=fontsize - 2,
                    color=median_text_color
                )

        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

    if title:
        fig.suptitle(title, fontsize=fontsize + 1)

    return fig, axes