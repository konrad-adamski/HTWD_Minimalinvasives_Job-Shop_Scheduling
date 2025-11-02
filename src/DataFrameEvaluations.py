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
            return "DEVIATION"
        elif value == "GT_DEVIATION_OPTIMIZE":
            return "DEVIATION_INSERT"
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
    x_label = x_col_as #or x_col
    y_label = y_col_as # or y_col
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
    xlabel_at_col: int = 0,
    ratio_label_on: bool = True,
    ymax: Optional[float] = None,
    # 🆕: Boxbreite und Text-Verschiebung
    box_width: float = 0.4,
    median_text_dx: float = 0.25,
    x_lim_extra: float = 0.6,
):
    value_label = value_as # or value_col
    x_label = x_col_as # or x_col
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

    boxprops = dict(linewidth=1.2, color="black")
    whiskerprops = dict(linewidth=1.1, color="black")
    capprops = dict(linewidth=1.1, color="black")
    medianprops = dict(linewidth=1.5, color=median_text_color)
    flierprops = dict(
        marker='o', markersize=6, markerfacecolor='none',
        markeredgecolor='gray', alpha=0.6
    )

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
            widths=box_width,             # 🆕 schlankere Boxen
            showfliers=flier_visible,
            patch_artist=False,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops,
            flierprops=flierprops,
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [ratio_label(x, ratio_label_on) for x in x_levels],
            fontsize=fontsize - 1
        )

        if j == xlabel_at_col:
            ax.set_xlabel(x_label, fontsize=fontsize)
        else:
            ax.set_xlabel("")

        if j == 0:
            ax.set_ylabel(value_label, fontsize=fontsize)

        if col_col is not None:
            ax.set_title(
                f"{col_label} = {c}",
                fontsize=fontsize,
                bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
                pad=11
            )

        # 🆕 Median-Werte RECHTS neben der Box anzeigen
        if show_median_labels:
            for k, med_line in enumerate(bp["medians"]):
                # x-Mitte des Medianstrichs
                xm = float(np.mean(med_line.get_xdata()))
                # y-Höhe des Medianstrichs
                ym = float(np.mean(med_line.get_ydata()))
                ax.text(
                    xm + median_text_dx,     # leicht rechts
                    ym,
                    format(ym, median_fmt),
                    ha="left", va="center",  # mittig zur Linie
                    fontsize=fontsize - 2,
                    color=median_text_color,
                )

        if ymax is not None:
            ax.set_ylim(top=ymax)

        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

        ax.set_xlim(right=len(x_levels) + x_lim_extra)

    if title:
        fig.suptitle(title, fontsize=fontsize + 1)

    return fig, axes


def plot_experiment_lines_compare(
    df_values: pd.DataFrame,
    df_meta: pd.DataFrame,
    *,
    value_col: str,
    id_col: str = "Experiment_ID",
    x_col: str = "Inner Tardiness Ratio",
    compare_col: str = "Max Bottleneck Utilization",
    value_as: Optional[str] = None,
    x_col_as: Optional[str] = None,
    compare_col_as: Optional[str] = None,
    agg_method: Literal["mean", "median"] = "mean",
    show_quantile_band: bool = True,
    quantile_band: Tuple[float, float] = (0.25, 0.75),
    figsize: Tuple[float, float] = (7.0, 4.2),
    linewidth: float = 2.0,
    marker: Optional[str] = "o",
    markersize: float = 4.5,
    alpha_line: float = 0.95,
    alpha_band: float = 0.20,
    fontsize: int = 12,
    title: Optional[str] = None,
    grid: bool = True,
    compare_col_is_ratio: bool = False,
    dodge: float = 0.04,
    use_distinct_linestyles: bool = True,
    marker_edgecolor: str = "white",
    marker_edgewidth: float = 0.8,
    ratio_label_on: bool = True,
    # ✅ Neu:
    ymax: Optional[float] = None,
    ymin: Optional[float] = None,
    legend_loc: Literal[
        "best", "upper left", "upper right", "lower left", "lower right",
        "center"
    ] = "best"
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

    # Offsets (dodge)
    base_positions = np.arange(n_cmp) - (n_cmp - 1) / 2.0
    offsets = (base_positions * dodge)
    offset_map = {cl: offsets[i] for i, cl in enumerate(cmp_levels)}
    max_off = abs(offsets).max() if n_cmp > 0 else 0.0

    # Quantilbänder zeichnen
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

    # Linien + Marker
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

    # Achsen / Labels
    ax.set_xticks(np.arange(len(x_levels)))
    ax.set_xticklabels([ratio_label(x, ratio_label_on) for x in x_levels], fontsize=fontsize-1)
    ax.set_xlim(-0.5 - max_off, (len(x_levels) - 0.5) + max_off)
    ax.set_xlabel(x_col_as, fontsize=fontsize)
    ax.set_ylabel(value_as, fontsize=fontsize)

    # ✅ manuelles Y-Limit
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    # Legende, Grid, Titel
    leg_title = compare_col_as or compare_col
    #ax.legend(title=leg_title, fontsize=fontsize-1, title_fontsize=fontsize-1, frameon=True)
    leg = ax.legend(
        title=leg_title,
        fontsize=fontsize - 1,
        title_fontsize=fontsize - 1,
        frameon=True,
        loc=legend_loc,
    )
    leg.get_frame().set_alpha(0.70)

    if grid:
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7, zorder=0)
    if title:
        ax.set_title(title, fontsize=fontsize+1)

    return fig, ax

def compute_experiment_summary_df(
    df_values: pd.DataFrame,
    df_meta: pd.DataFrame,
    *,
    value_col: str,
    value_as: Optional[str] = None,
    id_col: str = "Experiment_ID",
    axis_a: str = "Inner Tardiness Ratio",       # ehemals x_col
    axis_b: str = "Max Bottleneck Utilization",  # ehemals compare_col
    axis_a_as: Optional[str] = None,             # ✅ neuer Anzeigename für Axis A
    axis_b_as: Optional[str] = None,             # ✅ neuer Anzeigename für Axis B
    agg_method: Literal["mean", "median"] = "mean",
    show_quantile_band: bool = True,
    quantile_band: Tuple[float, float] = (0.25, 0.75),
    sort_levels: bool = True,
    round_digits: Optional[int] = None,          # Rundung
    round_axes: bool = False,                    # Achsen runden? (Default: nur Werte)
) -> pd.DataFrame:
    """
    Aggregiert `value_col` je (axis_b, axis_a) und liefert ein DataFrame:
    [AxisBLabel, AxisALabel, <value_as>, <value_as>_Qxx, <value_as>_Qyy]

    - Achsenspalten heißen per Default wie `axis_a`/`axis_b`, oder wie
      `axis_a_as`/`axis_b_as`, wenn angegeben.
    - Ergebnis-Spalten erhalten Namen aus `value_as` (oder `value_col`) + Quantil-Suffixe.
    - Optional Rundung der Ergebnis-Spalten (und auf Wunsch der Achsen).
    """

    # 1) Validierung
    if id_col not in df_values.columns:
        raise ValueError(f"`df_values` braucht die Spalte `{id_col}`.")
    for needed in [id_col, axis_a, axis_b]:
        if needed not in df_meta.columns:
            raise ValueError(f"`df_meta` fehlt Spalte `{needed}`.")
    if agg_method not in {"mean", "median"}:
        raise ValueError("agg_method muss 'mean' oder 'median' sein.")

    # 2) Merge
    meta_cols = [id_col, axis_a, axis_b]
    dfm = df_values.merge(df_meta[meta_cols], on=id_col, how="left")

    # 3) Sortierung/ Kategorien
    if sort_levels:
        a_levels = np.sort(dfm[axis_a].dropna().unique())
        b_levels = np.sort(dfm[axis_b].dropna().unique())
        dfm[axis_a] = pd.Categorical(dfm[axis_a], categories=a_levels, ordered=True)
        dfm[axis_b] = pd.Categorical(dfm[axis_b], categories=b_levels, ordered=True)

    # 4) Aggregation (mean/median)
    agg_name = value_as or value_col
    group_keys = [axis_b, axis_a]

    if agg_method == "mean":
        df_line = (
            dfm.groupby(group_keys, as_index=False, observed=True)[value_col]
               .mean()
               .rename(columns={value_col: agg_name})
        )
    else:
        df_line = (
            dfm.groupby(group_keys, as_index=False, observed=True)[value_col]
               .median()
               .rename(columns={value_col: agg_name})
        )

    # 5) Quantile
    if show_quantile_band:
        q_lo, q_hi = quantile_band
        if not (0.0 <= q_lo < q_hi <= 1.0):
            raise ValueError("quantile_band muss 0.0 <= q_lo < q_hi <= 1.0 erfüllen.")
        df_q = (
            dfm.groupby(group_keys, observed=True)[value_col]
               .quantile([q_lo, q_hi])
               .unstack(level=-1)
               .reset_index()
               .rename(columns={
                   q_lo: f"{agg_name}_Q{int(q_lo*100)}",
                   q_hi: f"{agg_name}_Q{int(q_hi*100)}"
               })
        )
    else:
        df_q = pd.DataFrame(columns=group_keys)

    # 6) Merge Linie + Quantile
    df_out = df_line.merge(df_q, on=group_keys, how="left")

    # 7) Spalten labeln
    col_a = axis_a_as or axis_a
    col_b = axis_b_as or axis_b
    df_out = df_out.rename(columns={axis_a: col_a, axis_b: col_b})

    # 8) Rundung
    if round_digits is not None:
        # Ergebnis-Spalten (immer runden)
        result_cols = [agg_name] + [c for c in df_out.columns if c.startswith(f"{agg_name}_Q")]
        result_cols = [c for c in result_cols if c in df_out.columns]
        if result_cols:
            df_out[result_cols] = df_out[result_cols].round(round_digits)
        # Optional: Achsen ebenfalls runden (falls numerisch)
        if round_axes:
            axis_numeric = [c for c in [col_a, col_b] if c in df_out.columns and np.issubdtype(df_out[c].dtype, np.number)]
            if axis_numeric:
                df_out[axis_numeric] = df_out[axis_numeric].round(round_digits)

    # 9) Sortierung & Return
    return df_out.sort_values([col_b, col_a]).reset_index(drop=True)