import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools

from matplotlib.colors import Normalize
from typing import Optional, Tuple, Literal
from fractions import Fraction

ExtendType = Literal["auto", "both", "min", "max", None]
AggMethod = Literal["mean", "median"]

def ratio_label(value: float, ratio_label_on: bool = True, max_denominator: int = 10) -> str:
    if ratio_label_on is False:
        if value == "GT_DEVIATION":
            return "DEV"
        elif value == "GT_DEVIATION_OPTIMIZE":
            return "DEV 'optimize'"
        elif value == "GT_SLACK":
            return "SLACK"
        return value
    frac = Fraction(value).limit_denominator(max_denominator)
    t = frac.numerator
    e = frac.denominator - t
    return f"{t}:{e}"


def plot_experiment_heatmaps(
        df_values: pd.DataFrame,    # z.B. df_shift_dev (Verteilungen je Shift; enthält id_col + value_col)
        df_meta: pd.DataFrame,      # z.B. df_experiments (Parameter je Experiment_ID; enthält id_col + x/y/col/row)
        *,
        value_col: str,             # z.B. "Deviation"
        id_col: str = "Experiment_ID",
        x_col: str = "Inner Tardiness Ratio",
        y_col: str = "Abs Lateness Ratio",
        col_col: Optional[str] = None,
        row_col: Optional[str] = None,
        # Anzeigenamen (Labels)
        value_as: Optional[str] = None,
        x_col_as: Optional[str] = None,
        y_col_as: Optional[str] = None,
        col_col_as: Optional[str] = None,
        row_col_as: Optional[str] = None,
        # Darstellung
        agg_method: AggMethod = "mean",
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
        # In welcher Spalten-Facette das x-Achsenlabel gezeigt wird (0-basiert)
        xlabel_at_col: int = 0,
        ratio_label_on: bool = True,
):
    # 0) Labels
    value_label = value_as or value_col
    x_label = x_col_as or x_col
    y_label = y_col_as or y_col
    col_label = col_col_as or (col_col if col_col is not None else "")
    row_label = row_col_as or (row_col if row_col is not None else "")

    # 1) Merge Werte + Meta
    needed_meta = [id_col, x_col, y_col] + ([col_col] if col_col else []) + ([row_col] if row_col else [])
    missing_vals = [id_col, value_col]
    for c in missing_vals:
        if c not in df_values.columns:
            raise ValueError(f"`df_values` fehlt Spalte `{c}`.")
    for c in needed_meta:
        if c and c not in df_meta.columns:
            raise ValueError(f"`df_meta` fehlt Spalte `{c}`.")

    dfm = df_values[[id_col, value_col]].merge(df_meta[needed_meta], on=id_col, how="left")

    # 2) Facetten-Keys (optional)
    unique_cols = [None] if col_col is None else sorted(dfm[col_col].dropna().unique())
    unique_rows = [None] if row_col is None else sorted(dfm[row_col].dropna().unique())
    n_cols, n_rows = len(unique_cols), len(unique_rows)

    # xlabel_at_col bounds
    if n_cols > 1:
        xlabel_at_col = int(np.clip(xlabel_at_col, 0, n_cols - 1))
    else:
        xlabel_at_col = 0

    # 3) Diskrete Stufen (über alle Daten)
    x_levels = list(np.sort(dfm[x_col].dropna().unique()))
    y_levels = list(np.sort(dfm[y_col].dropna().unique()))

    # 4) Aggregation vorbereiten: pro (row?, col?, y, x)
    group_keys = [x_col, y_col]
    if col_col is not None: group_keys.append(col_col)
    if row_col is not None: group_keys.append(row_col)

    if agg_method == "mean":
        df_agg = (dfm.groupby(group_keys, as_index=False)[value_col].mean())
    else:
        df_agg = (dfm.groupby(group_keys, as_index=False)[value_col].median())

    # 5) Wertebereich aus Aggregaten
    z_all = df_agg[value_col].to_numpy(dtype=float)
    data_min, data_max = float(np.nanmin(z_all)), float(np.nanmax(z_all))
    if vmin is None: vmin = data_min
    if vmax is None: vmax = data_max
    if vmin > vmax:
        raise ValueError("vmin darf nicht größer als vmax sein.")

    # 6) Colormap + ggf. Richtungsumkehr
    base_cmap = mpl.colormaps.get_cmap(cmap_name)
    cmap = base_cmap
    if auto_reverse_cmap:
        ends_with_r = cmap_name.endswith("_r")
        want_reversed = not higher_is_better
        if want_reversed ^ ends_with_r:
            cmap = base_cmap.reversed()
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    # 7) Figure/Axes
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_scale[0] * n_cols, figsize_scale[1] * n_rows),
        constrained_layout=True, sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)

    # 8) Facetten zeichnen
    for i, r in enumerate(unique_rows):
        for j, c in enumerate(unique_cols):
            ax = axes[i, j]
            sub = df_agg.copy()
            if row_col is not None:
                sub = sub[sub[row_col] == r]
            if col_col is not None:
                sub = sub[sub[col_col] == c]

            if sub.empty:
                ax.set_visible(False)
                continue

            # Pivot auf Stufenraster
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

            # Ticks
            ax.set_xticks(np.arange(nx) + 0.5)
            ax.set_yticks(np.arange(ny) + 0.5)
            ax.set_xticklabels([ratio_label(x, ratio_label_on) for x in x_levels])
            ax.set_yticklabels([ratio_label(y, ratio_label_on) for y in y_levels])

            # Spaltentitel
            if i == 0 and col_col is not None:
                ax.set_title(
                    f"{col_label} = {c}",
                    fontsize=12,
                    pad=10,
                    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
                )

            # Y-Label links
            if j == 0:
                if row_col is not None:
                    ax.text(
                        -0.25, 0.5,
                        f"{row_label} = {r}",
                        transform=ax.transAxes,
                        fontsize=12,
                        ha="center", va="center",
                        rotation=90,
                        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
                    )
                    ax.set_ylabel(y_label, fontsize=12, labelpad=15)
                else:
                    ax.set_ylabel(y_label)

            # x-Achsenlabel nur in gewünschter Spalte
            if j == xlabel_at_col:
                ax.set_xlabel(x_label)
            else:
                ax.set_xlabel("")

            # Zellenwerte annotieren (optional)
            if annot:
                vals = pivot.values
                for yi in range(ny):
                    for xi in range(nx):
                        v = vals[yi, xi]
                        if np.isfinite(v):
                            ax.text(
                                xi + 0.5, yi + 0.5, format(v, fmt),
                                ha="center", va="center", color=text_color, fontsize=fontsize
                            )

    # 9) Colorbar + extend
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

    # 10) Supertitel
    if title is not None:
        fig.suptitle(title, fontsize=fontsize)

    return fig, axes


# Wrapper: Kendall Tau (hoch = gut)
def plot_experiment_heatmaps_kendall_tau(
        df_values: pd.DataFrame,
        df_meta: pd.DataFrame,
        *,
        value_col: str, value_as: str = "Kendall τ",
        id_col: str = "Experiment_ID",
        x_col: str = "Inner Tardiness Ratio", x_col_as: Optional[str] = None,
        y_col: str = "Abs Lateness Ratio", y_col_as: Optional[str] = None,
        col_col: Optional[str] = None, col_col_as: Optional[str] = None,
        row_col: Optional[str] = None, row_col_as: Optional[str] = None,
        vmin: Optional[float] = 0.7,
        vmax: Optional[float] = 1.0,
        fmt: str = ".2f",
        extend: ExtendType = "both",
        title: Optional[str] = None,
        fontsize: int = 13,
        legend_steps: int = 6,
        xlabel_at_col: int = 0,
        agg_method: AggMethod = "mean",
):
    return plot_experiment_heatmaps(
        df_values=df_values,
        df_meta=df_meta,
        value_col=value_col, id_col=id_col,
        x_col=x_col, y_col=y_col,
        col_col=col_col, row_col=row_col,
        value_as=value_as,
        x_col_as=x_col_as, y_col_as=y_col_as,
        col_col_as=col_col_as, row_col_as=row_col_as,
        agg_method=agg_method,
        vmin=vmin, vmax=vmax,
        fmt=fmt, extend=extend,
        annot=True,
        cmap_name="RdYlGn",
        higher_is_better=True,
        auto_reverse_cmap=True,
        title=title,
        fontsize=fontsize,
        legend_steps=legend_steps,
        xlabel_at_col=xlabel_at_col,
    )


# Wrapper: niedrig = gut
def plot_experiment_heatmaps_good_low(
        df_values: pd.DataFrame,
        df_meta: pd.DataFrame,
        *,
        value_col: str, value_as: Optional[str] = None,
        id_col: str = "Experiment_ID",
        x_col: str = "Inner Tardiness Ratio",
        y_col: str = "Abs Lateness Ratio",
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
        xlabel_at_col: int = 0,
        agg_method: AggMethod = "mean",
        **kwargs
):
    return plot_experiment_heatmaps(
        df_values=df_values,
        df_meta=df_meta,
        value_col=value_col, id_col=id_col,
        x_col=x_col, y_col=y_col,
        col_col=col_col, row_col=row_col,
        value_as=value_as,
        x_col_as=x_col_as, y_col_as=y_col_as,
        col_col_as=col_col_as, row_col_as=row_col_as,
        agg_method=agg_method,
        vmin=vmin, vmax=vmax,
        fmt=fmt, extend=extend,
        cmap_name=cmap_name,
        higher_is_better=False,
        auto_reverse_cmap=True,
        title=title,
        xlabel_at_col=xlabel_at_col,
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
    ratio_label_on: bool = True,
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
        ax.set_xticklabels([ratio_label(x, ratio_label_on) for x in x_levels], fontsize=fontsize - 1)

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
                pad=11
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

def plot_experiment_lines_compare(
    df_values: pd.DataFrame,           # z.B. df_shift_dev (Verteilungen je Shift)
    df_meta: pd.DataFrame,             # z.B. df_experiments (Parameter je Experiment_ID)
    *,
    value_col: str,                    # z.B. "Deviation"
    id_col: str = "Experiment_ID",
    x_col: str = "Inner Tardiness Ratio",
    compare_col: str = "Max Bottleneck Utilization",
    # Labels:
    value_as: Optional[str] = None,
    x_col_as: Optional[str] = None,
    compare_col_as: Optional[str] = None,
    # Darstellung / Aggregation:
    agg_method: Literal["mean", "median"] = "mean",
    show_quantile_band: bool = True,
    quantile_band: Tuple[float, float] = (0.25, 0.75),  # (unteres, oberes Quantil)
    figsize: Tuple[float, float] = (7.0, 4.2),
    linewidth: float = 2.0,
    marker: Optional[str] = "o",
    markersize: float = 4.5,
    alpha_line: float = 0.95,
    alpha_band: float = 0.20,
    fontsize: int = 12,
    title: Optional[str] = None,
    grid: bool = True,
    # 🔹 NEU:
    compare_col_is_ratio: bool = False,
    ratio_label_on: bool = True
):
    """
    Zeichnet Linien-Kurven über x_col für jede Ausprägung von compare_col.
    Aggregation pro (compare_col, x_col) via agg_method; optional Quantilband.
    df_values muss `id_col` enthalten; df_meta enthält `x_col`/`compare_col` je Experiment.

    Parameter:
    ----------
    compare_col_is_ratio : bool
        Wenn True, wird `ratio_label()` auch auf die compare_col-Werte angewendet
        (z. B. "1:3" statt 0.333...).
    """
    # -- Validierung
    if id_col not in df_values.columns:
        raise ValueError(f"`df_values` braucht die Spalte `{id_col}`.")
    for needed in [id_col, x_col, compare_col]:
        if needed not in df_meta.columns:
            raise ValueError(f"`df_meta` fehlt Spalte `{needed}`.")

    # -- Merge Werte + Meta
    meta_cols = [id_col, x_col, compare_col]
    dfm = df_values.merge(df_meta[meta_cols], on=id_col, how="left")

    # X-Levels & Compare-Levels
    x_levels = list(np.sort(dfm[x_col].dropna().unique()))
    cmp_levels = list(np.sort(dfm[compare_col].dropna().unique()))

    # Aggregation (Linienwerte)
    if agg_method == "mean":
        df_line = (
            dfm.groupby([compare_col, x_col], as_index=False)[value_col]
               .mean()
               .rename(columns={value_col: "y"})
        )
    else:  # "median"
        df_line = (
            dfm.groupby([compare_col, x_col], as_index=False)[value_col]
               .median()
               .rename(columns={value_col: "y"})
        )

    # Optional: Quantilband vorbereiten
    if show_quantile_band:
        q_lo, q_hi = quantile_band
        if not (0.0 <= q_lo < q_hi <= 1.0):
            raise ValueError("quantile_band muss 0.0 <= q_lo < q_hi <= 1.0 erfüllen.")
        df_q = (
            dfm.groupby([compare_col, x_col])[value_col]
               .quantile([q_lo, q_hi])
               .unstack(level=-1)
               .reset_index()
               .rename(columns={q_lo: "y_lo", q_hi: "y_hi"})
        )
    else:
        df_q = None

    # Plot
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Für saubere Linien in fester x-Reihenfolge pivoten
    line_pivot = (
        df_line.pivot(index=x_col, columns=compare_col, values="y")
              .reindex(index=x_levels)
    )

    # Farben pro compare-Level (Matplotlib Zyklus)
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
    color_map = {}
    for idx, cl in enumerate(cmp_levels):
        color_map[cl] = colors[idx % len(colors)] if colors else None

    # Quantilbänder zeichnen (pro compare-Level)
    if df_q is not None:
        band_pivot_lo = (
            df_q.pivot(index=x_col, columns=compare_col, values="y_lo")
                .reindex(index=x_levels)
        )
        band_pivot_hi = (
            df_q.pivot(index=x_col, columns=compare_col, values="y_hi")
                .reindex(index=x_levels)
        )
        for cl in cmp_levels:
            ylo = band_pivot_lo[cl].to_numpy() if cl in band_pivot_lo else None
            yhi = band_pivot_hi[cl].to_numpy() if cl in band_pivot_hi else None
            if ylo is None or yhi is None:
                continue
            mask = np.isfinite(ylo) & np.isfinite(yhi)
            if not np.any(mask):
                continue
            xs = np.arange(len(x_levels))[mask]
            ax.fill_between(
                xs, ylo[mask], yhi[mask],
                alpha=alpha_band, linewidth=0, color=color_map[cl]
            )

    # Linien + Marker zeichnen
    for cl in cmp_levels:
        y = line_pivot[cl].to_numpy() if cl in line_pivot else None
        if y is None:
            continue
        xs = np.arange(len(x_levels))
        label_str = ratio_label(cl, ratio_label_on) if compare_col_is_ratio else str(cl)
        ax.plot(
            xs, y,
            marker=marker, markersize=markersize,
            linewidth=linewidth, alpha=alpha_line,
            label=label_str, color=color_map[cl]
        )

    # Achsen / Ticks / Labels
    ax.set_xticks(np.arange(len(x_levels)))
    ax.set_xticklabels([ratio_label(x, ratio_label_on) for x in x_levels], fontsize=fontsize-1)
    ax.set_xlabel(x_col_as or x_col, fontsize=fontsize)
    ax.set_ylabel(value_as or value_col, fontsize=fontsize)

    # Legende
    leg_title = compare_col_as or compare_col
    ax.legend(title=leg_title, fontsize=fontsize-1, title_fontsize=fontsize-1, frameon=True)

    # Grid
    if grid:
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

    # Titel
    if title:
        ax.set_title(title, fontsize=fontsize+1)

    return fig, ax


def plot_experiment_lines_compare(
    df_values: pd.DataFrame,           # z.B. df_shift_dev (Verteilungen je Shift)
    df_meta: pd.DataFrame,             # z.B. df_experiments (Parameter je Experiment_ID)
    *,
    value_col: str,                    # z.B. "Deviation"
    id_col: str = "Experiment_ID",
    x_col: str = "Inner Tardiness Ratio",
    compare_col: str = "Max Bottleneck Utilization",
    # Labels:
    value_as: Optional[str] = None,
    x_col_as: Optional[str] = None,
    compare_col_as: Optional[str] = None,
    # Darstellung / Aggregation:
    agg_method: Literal["mean", "median"] = "mean",
    show_quantile_band: bool = True,
    quantile_band: Tuple[float, float] = (0.25, 0.75),  # (unteres, oberes Quantil)
    figsize: Tuple[float, float] = (7.0, 4.2),
    linewidth: float = 2.0,
    marker: Optional[str] = "o",
    markersize: float = 4.5,
    alpha_line: float = 0.95,
    alpha_band: float = 0.20,
    fontsize: int = 12,
    title: Optional[str] = None,
    grid: bool = True,
    # Ratio-Labels für compare_col
    compare_col_is_ratio: bool = False,
    # 🔹 Neu gegen Überdeckung:
    dodge: float = 0.04,               # horizontaler Versatz pro compare-Linie (in „x-Level“-Einheiten)
    use_distinct_linestyles: bool = True,
    marker_edgecolor: str = "white",
    marker_edgewidth: float = 0.8,
    ratio_label_on: bool = True
):
    """
    Linien-Plot über x_col, je compare_col eine Linie.
    Aggregation pro (compare_col, x_col) via agg_method; optional Quantilband.
    Überdeckung wird reduziert via horizontalem Versatz (dodge) und optionalen Linienstilen.
    """
    # -- Validierung
    if id_col not in df_values.columns:
        raise ValueError(f"`df_values` braucht die Spalte `{id_col}`.")
    for needed in [id_col, x_col, compare_col]:
        if needed not in df_meta.columns:
            raise ValueError(f"`df_meta` fehlt Spalte `{needed}`.")

    # -- Merge Werte + Meta
    meta_cols = [id_col, x_col, compare_col]
    dfm = df_values.merge(df_meta[meta_cols], on=id_col, how="left")

    # X-Levels & Compare-Levels
    x_levels = list(np.sort(dfm[x_col].dropna().unique()))
    cmp_levels = list(np.sort(dfm[compare_col].dropna().unique()))
    n_cmp = len(cmp_levels)

    # Aggregation (Linienwerte)
    if agg_method == "mean":
        df_line = (
            dfm.groupby([compare_col, x_col], as_index=False)[value_col]
               .mean()
               .rename(columns={value_col: "y"})
        )
    else:  # "median"
        df_line = (
            dfm.groupby([compare_col, x_col], as_index=False)[value_col]
               .median()
               .rename(columns={value_col: "y"})
        )

    # Optional: Quantilband
    if show_quantile_band:
        q_lo, q_hi = quantile_band
        if not (0.0 <= q_lo < q_hi <= 1.0):
            raise ValueError("quantile_band muss 0.0 <= q_lo < q_hi <= 1.0 erfüllen.")
        df_q = (
            dfm.groupby([compare_col, x_col])[value_col]
               .quantile([q_lo, q_hi])
               .unstack(level=-1)
               .reset_index()
               .rename(columns={q_lo: "y_lo", q_hi: "y_hi"})
        )
    else:
        df_q = None

    # Plot
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Pivot für feste x-Reihenfolge
    line_pivot = (
        df_line.pivot(index=x_col, columns=compare_col, values="y")
              .reindex(index=x_levels)
    )

    # Farben & Linienstile
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', None) or [None]
    linestyles = itertools.cycle(["-", "--", "-.", ":"]) if use_distinct_linestyles else itertools.cycle(["-"])
    color_map = {cl: colors[i % len(colors)] for i, cl in enumerate(cmp_levels)}
    linestyle_map = {cl: next(linestyles) for cl in cmp_levels}

    # Offsets (dodge) rund um die Mitte verteilen
    # Beispiel: n=1 -> [0]; n=2 -> [-0.5, +0.5]*dodge; n=3 -> [-1,0,1]*dodge; ...
    base_positions = np.arange(n_cmp) - (n_cmp - 1) / 2.0
    offsets = (base_positions * dodge)
    offset_map = {cl: offsets[i] for i, cl in enumerate(cmp_levels)}
    max_off = abs(offsets).max() if n_cmp > 0 else 0.0

    # Quantilbänder zeichnen (mit Offset)
    if df_q is not None:
        band_pivot_lo = (
            df_q.pivot(index=x_col, columns=compare_col, values="y_lo")
                .reindex(index=x_levels)
        )
        band_pivot_hi = (
            df_q.pivot(index=x_col, columns=compare_col, values="y_hi")
                .reindex(index=x_levels)
        )
        for cl in cmp_levels:
            if cl not in band_pivot_lo or cl not in band_pivot_hi:
                continue
            ylo = band_pivot_lo[cl].to_numpy()
            yhi = band_pivot_hi[cl].to_numpy()
            mask = np.isfinite(ylo) & np.isfinite(yhi)
            if not np.any(mask):
                continue
            xs_base = np.arange(len(x_levels))
            xs = xs_base + offset_map[cl]
            ax.fill_between(
                xs[mask], ylo[mask], yhi[mask],
                alpha=alpha_band, linewidth=0, color=color_map[cl]
            )

    # Linien + Marker (mit Offset)
    for cl in cmp_levels:
        if cl not in line_pivot:
            continue
        y = line_pivot[cl].to_numpy()
        xs_base = np.arange(len(x_levels))
        xs = xs_base + offset_map[cl]
        label_str = ratio_label(cl, ratio_label_on) if compare_col_is_ratio else str(cl)
        ax.plot(
            xs, y,
            marker=marker, markersize=markersize,
            markeredgecolor=marker_edgecolor, markeredgewidth=marker_edgewidth,
            linewidth=linewidth, alpha=alpha_line,
            linestyle=linestyle_map[cl],
            label=label_str, color=color_map[cl],
            zorder=3
        )

    # Achsen / Ticks / Labels
    ax.set_xticks(np.arange(len(x_levels)))
    ax.set_xticklabels([ratio_label(x, ratio_label_on) for x in x_levels], fontsize=fontsize-1)
    ax.set_xlim(-0.5 - max_off, (len(x_levels) - 0.5) + max_off)

    ax.set_xlabel(x_col_as or x_col, fontsize=fontsize)
    ax.set_ylabel(value_as or value_col, fontsize=fontsize)

    # Legende
    leg_title = compare_col_as or compare_col
    ax.legend(title=leg_title, fontsize=fontsize-1, title_fontsize=fontsize-1, frameon=True)

    # Grid
    if grid:
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7, zorder=0)

    # Titel
    if title:
        ax.set_title(title, fontsize=fontsize+1)

    return fig, ax

