import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def proof_of_concept_v2(dev_A, dev_B,
                        tardiness_A, earliness_A,
                        tardiness_B, earliness_B,
                        label_A="Strategie A",
                        label_B="Strategie B",
                        title="Rescheduling-Vergleich",
                        ylabel_left="Startzeitabweichung (Minuten)",
                        ylabel_right="Termintreue",
                        y_right_lim_min=None,
                        y_right_lim_max=None,
                        as_percentage=True):

    days = np.arange(len(dev_A))
    assert len(dev_A) == len(dev_B) == len(tardiness_A) == len(tardiness_B), "Alle Listen müssen gleich lang sein."

    # Farben
    color_bar_A = 'darkblue'
    color_bar_B = 'darkred'
    color_tard_A = 'blue'
    color_early_A = '#87CEEB'
    color_tard_B = 'red'
    color_early_B = '#FF6347'

    bar_width = 0.4

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Balken
    bars_A = ax1.bar(days - bar_width/2, dev_A, width=bar_width, label=f"{label_A} – Abweichung", color=color_bar_A)
    bars_B = ax1.bar(days + bar_width/2, dev_B, width=bar_width, label=f"{label_B} – Abweichung", color=color_bar_B)

    ax1.set_xlabel("Tag")
    ax1.set_ylabel(ylabel_left, color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    for bar in bars_A:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.1f}",
                 ha='center', va='bottom', fontsize=9, color=color_bar_A)

    for bar in bars_B:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.1f}",
                 ha='center', va='bottom', fontsize=9, color=color_bar_B)

    # Zweite y-Achse (Linien)
    ax2 = ax1.twinx()
    factor = 100 if as_percentage else 1
    unit = "%" if as_percentage else "Minuten"

    t_A = np.array(tardiness_A) * factor
    e_A = -np.array(earliness_A) * factor
    t_B = np.array(tardiness_B) * factor
    e_B = -np.array(earliness_B) * factor

    ax2.plot(days, t_A, marker='o', color=color_tard_A, label=f"{label_A} – Tardiness")
    ax2.plot(days, e_A, marker='s', color=color_early_A, label=f"{label_A} – Earliness")
    ax2.plot(days, t_B, marker='o', linestyle='--', color=color_tard_B, label=f"{label_B} – Tardiness")
    ax2.plot(days, e_B, marker='s', linestyle='--', color=color_early_B, label=f"{label_B} – Earliness")

    ax2.set_ylabel(f"{ylabel_right} ({unit})", color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    if y_right_lim_min is not None and y_right_lim_max is not None:
        ax2.set_ylim(y_right_lim_min, y_right_lim_max)
    elif y_right_lim_min is not None:
        ax2.set_ylim(bottom=y_right_lim_min)
    elif y_right_lim_max is not None:
        ax2.set_ylim(top=y_right_lim_max)



    #def label_line(x, y, color):
    #    for xi, yi in zip(x, y):
    #        if abs(yi) > 0.01:
    #            ax2.text(xi + 0.1, yi + 0.5 * np.sign(yi), f"{yi:.1f}",
    #                     ha='center', va='bottom', fontsize=8, color=color)
    def label_line(x, y, color):
        for xi, yi in zip(x, y):
            ax2.text(xi + 0.1, yi + 0.5, f"{yi:.1f}", ha='center', va='bottom', fontsize=8, color=color)

    label_line(days, t_A, color_tard_A)
    label_line(days, e_A, color_early_A)
    label_line(days, t_B, color_tard_B)
    label_line(days, e_B, color_early_B)

    plt.title(title)
    ax1.set_xticks(days)
    ax1.set_xticklabels([str(d + 1) for d in days])
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Gemeinsame Legende
    lines_labels = ax1.get_legend_handles_labels()
    lines2_labels = ax2.get_legend_handles_labels()
    all_lines = lines_labels[0] + lines2_labels[0]
    all_labels = lines_labels[1] + lines2_labels[1]
    ax1.legend(all_lines, all_labels, loc='upper left', bbox_to_anchor=(1.04, 1))

    fig.tight_layout()
    plt.show()


def proof_of_concept_v1(dev_A, dev_B,
                        tardiness_A, earliness_A,
                        tardiness_B, earliness_B,
                        label_A="Strategie A",
                        label_B="Strategie B",
                        title="Proof of Concept: Rescheduling-Vergleich",
                        ylabel_left="Startzeitabweichung (Minuten)",
                        ylabel_right="Tardiness / Earliness (in %)",
                        y_right_lim=50):
    
    days = np.arange(len(dev_A))
    assert len(dev_A) == len(dev_B) == len(tardiness_A) == len(tardiness_B), "Alle Listen müssen gleich lang sein."

    # Farben
    color_bar_A = 'darkblue'
    color_bar_B = 'darkred'
    color_tard_A = 'blue'
    color_early_A = '#87CEEB'
    color_tard_B = 'red'
    color_early_B = '#FF6347'

    bar_width = 0.4

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Balken
    bars_A = ax1.bar(days - bar_width/2, dev_A, width=bar_width, label=f"{label_A} – Abweichung", color=color_bar_A)
    bars_B = ax1.bar(days + bar_width/2, dev_B, width=bar_width, label=f"{label_B} – Abweichung", color=color_bar_B)

    ax1.set_xlabel("Tag")
    ax1.set_ylabel(ylabel_left, color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    for bar in bars_A:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.1f}",
                 ha='center', va='bottom', fontsize=9, color=color_bar_A)

    for bar in bars_B:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.1f}",
                 ha='center', va='bottom', fontsize=9, color=color_bar_B)

    # Zweite y-Achse (Linien)
    ax2 = ax1.twinx()
    t_A = np.array(tardiness_A) * 100
    e_A = -np.array(earliness_A) * 100
    t_B = np.array(tardiness_B) * 100
    e_B = -np.array(earliness_B) * 100

    ax2.plot(days, t_A, marker='o', color=color_tard_A, label=f"{label_A} – Tardiness")
    ax2.plot(days, e_A, marker='s', color=color_early_A, label=f"{label_A} – Earliness")
    ax2.plot(days, t_B, marker='o', linestyle='--', color=color_tard_B, label=f"{label_B} – Tardiness")
    ax2.plot(days, e_B, marker='s', linestyle='--', color=color_early_B, label=f"{label_B} – Earliness")

    ax2.set_ylabel(ylabel_right, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(-y_right_lim, y_right_lim)

    def label_line(x, y, color):
        for xi, yi in zip(x, y):
            ax2.text(xi + 0.1, yi + 0.5, f"{yi:.1f}", ha='center', va='bottom', fontsize=8, color=color)

    label_line(days, t_A, color_tard_A)
    label_line(days, e_A, color_early_A)
    label_line(days, t_B, color_tard_B)
    label_line(days, e_B, color_early_B)

    plt.title(title)
    ax1.set_xticks(days)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Gemeinsame Legende
    lines_labels = ax1.get_legend_handles_labels()
    lines2_labels = ax2.get_legend_handles_labels()
    all_lines = lines_labels[0] + lines2_labels[0]
    all_labels = lines_labels[1] + lines2_labels[1]
    ax1.legend(all_lines, all_labels, loc='upper left', bbox_to_anchor=(1.04, 1))

    fig.tight_layout()
    plt.show()


# Combi --------------------------------------------------------------------------------------------------------------


def plot_tardiness_earliness_two_methods(tardiness_A, earliness_A,
                                         tardiness_B, earliness_B,
                                         labels=("Simple", "DevPen"),
                                         title="Vergleich der Termintreue",
                                         subtitle=None,
                                         ylabel="Ø Abweichung (Minuten)",
                                         y_lim_min=None, y_lim_max=None, as_percentage=False):
    days = np.arange(len(tardiness_A))

    # Optional in Prozent umrechnen
    factor = 100 if as_percentage else 1

    t_A = np.array(tardiness_A) * factor
    e_A = -np.array(earliness_A) * factor
    t_B = np.array(tardiness_B) * factor
    e_B = -np.array(earliness_B) * factor

    plt.figure(figsize=(10, 6))

    # Linien zeichnen
    plt.plot(days, t_A, marker='o', color='blue', label=f"{labels[0]} – Tardiness")
    plt.plot(days, e_A, marker='s', color='#87CEEB', label=f"{labels[0]} – Earliness")
    plt.plot(days, t_B, marker='o', linestyle='--', color='red', label=f"{labels[1]} – Tardiness")
    plt.plot(days, e_B, marker='s', linestyle='--', color='#FF6347', label=f"{labels[1]} – Earliness")

    # Werte beschriften
    def label_line(x, y):
        for xi, yi in zip(x, y):
            if abs(yi) > 0.01:
                plt.text(xi + 0.1, yi + 0.5 * np.sign(yi), f"{yi:.1f}", 
                         ha='center', va='bottom', fontsize=8)

    label_line(days, t_A)
    label_line(days, e_A)
    label_line(days, t_B)
    label_line(days, e_B)

    plt.xlabel("Zeit (in Tagen)")
    plt.ylabel(ylabel)
    
    full_title = title
    if subtitle:
        full_title = f"{title} {subtitle}"
    plt.title(full_title)

    plt.xticks(days)
    plt.grid(True, axis='y')

    # Y-Achsenlimits robust setzen
    if y_lim_min is not None and y_lim_max is not None:
        plt.ylim(y_lim_min, y_lim_max)
    elif y_lim_min is not None:
        plt.ylim(bottom=y_lim_min)
    elif y_lim_max is not None:
        plt.ylim(top=y_lim_max)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def plot_two_starttime_deviation_bars(dev_A, dev_B,
                                      label_A="Strategie A",
                                      label_B="Strategie B",
                                      title="Vergleich der Startzeitabweichungen jeder Operation pro Tag",
                                      ylabel="Summe der Abweichungen",
                                      xlabel="Tag"):
    days = list(range(len(dev_A)))
    assert len(dev_A) == len(dev_B), "Beide Deviation-Listen müssen gleich lang sein."

    bar_width = 0.4

    plt.figure(figsize=(10, 6))
    bars_A = plt.bar([d - bar_width/2 for d in days], dev_A, width=bar_width, label=label_A, color='darkblue')
    bars_B = plt.bar([d + bar_width/2 for d in days], dev_B, width=bar_width, label=label_B, color='darkred')

    for bar in bars_A:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.1f}",
                 ha='center', va='bottom', fontsize=9)

    for bar in bars_B:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.1f}",
                 ha='center', va='bottom', fontsize=9)

    plt.xlabel(xlabel)
    plt.ylabel(f"{ylabel} (in Minuten)")
    plt.title(title)
    plt.xticks(days)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


# Legacy -------------------------------------------------------------------------------------------------------------


def plot_tardiness_earliness_ideal_per_day(list_tardiness_count, list_earliness_count, list_ideal=None,
                                                title="Termintreue bei Job-Ende (Final Operations)",
                                                subtitle=None,
                                                ylabel="Anteil (in %)",  y_lim = 106):
    import matplotlib.pyplot as plt
    import numpy as np

    days = np.arange(len(list_tardiness_count))

    # Umrechnung Anteil → Prozent
    values1 = np.array(list_tardiness_count) * 100
    values2 = np.array(list_earliness_count) * 100
    values3 = np.array(list_ideal) * 100 if list_ideal is not None else None

    plt.figure(figsize=(10, 6))

    plt.plot(days, values1, marker='o', label="Tardiness > 0")
    plt.plot(days, values2, marker='s', label="Earliness > 0")
    if values3 is not None:
        plt.plot(days, values3, marker='^', label="Ideal (T=0 & E=0)")

    # Werte beschriften
    def label_points(x, y):
        for xi, yi in zip(x, y):
            plt.text(xi, yi + 1, f"{yi:.1f}", ha='center', va='bottom', fontsize=9)

    label_points(days, values1)
    label_points(days, values2)
    if values3 is not None:
        label_points(days, values3)

    plt.xlabel("Zeit (in Tagen)")
    plt.ylabel(ylabel)
    if subtitle:
        title = f"{title} – {subtitle}"
    plt.title(title)
    plt.xticks(days)

    # Legende
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.grid(True, axis='y')
    plt.ylim(0, y_lim)
    plt.tight_layout()
    plt.show()


def compare_and_plot_tardiness_single_day(df_a, df_b,
                               label_a='simple regeneration',
                               label_b='rescheduling with deviation penalty',
                               col_job='Job',
                               col_arrival='Arrival',
                               col_tardiness='Tardiness',
                               y_step=60,
                               figsize=(16, 8),
                               rotation=90,
                               show_regression=True):
    color_a = 'tab:blue'
    color_b = 'tab:olive'
    color_a_dark = 'darkblue'
    color_b_dark = 'tab:green'

    # Spaltennamen dynamisch aus Labels
    col_tardiness_a = f'Tardiness_{label_a.replace(" ", "_")}'
    col_tardiness_b = f'Tardiness_{label_b.replace(" ", "_")}'

    # Umbenennen
    df_a = df_a.rename(columns={col_tardiness: col_tardiness_a})
    df_b = df_b.rename(columns={col_tardiness: col_tardiness_b})

    # Merge
    df_compare = pd.merge(df_a, df_b, on=col_job, how='outer', suffixes=('_x', '_y'))
    df_compare['Arrival'] = df_compare[[f'{col_arrival}_x', f'{col_arrival}_y']].max(axis=1, skipna=True)
    df_compare.drop(columns=[f'{col_arrival}_x', f'{col_arrival}_y'], inplace=True)

    # Sortieren
    df_sorted = df_compare.sort_values(by='Arrival').reset_index(drop=True)
    x = np.arange(len(df_sorted)).reshape(-1, 1)
    jobs = df_sorted[col_job]

    # === Spezialbehandlung fehlender Werte ===
    mid_index = len(df_sorted) // 2
    mask_nan = df_sorted[col_tardiness_a].isna() | df_sorted[col_tardiness_b].isna()

    # Erste Hälfte: ersetze NaNs durch 0
    first_half = df_sorted.iloc[:mid_index].copy()
    first_half.loc[:, [col_tardiness_a, col_tardiness_b]] = \
        first_half[[col_tardiness_a, col_tardiness_b]].fillna(0)

    # Zweite Hälfte: entferne Zeilen mit NaN (nach Anzeige)
    second_half = df_sorted.iloc[mid_index:].copy()
    nan_rows = second_half[mask_nan.iloc[mid_index:]]

    if not nan_rows.empty:
        print("Entfernte Zeilen mit NaN in der zweiten Hälfte:")
        print(nan_rows[[col_job, col_tardiness_a, col_tardiness_b]])

    second_half = second_half[~mask_nan.iloc[mid_index:]]

    # Zusammenfügen
    df_sorted = pd.concat([first_half, second_half], ignore_index=True)
    x = np.arange(len(df_sorted)).reshape(-1, 1)
    jobs = df_sorted[col_job]

    # Y-Achsen-Ticks
    y_max = max(df_sorted[col_tardiness_a].max(), df_sorted[col_tardiness_b].max())
    y_ticks = np.arange(0, y_max + y_step, y_step)

    # Plot
    plt.figure(figsize=figsize)
    plt.plot(x, df_sorted[col_tardiness_a], marker='.', linestyle='--', label=label_a, color=color_a)
    plt.plot(x, df_sorted[col_tardiness_b], marker='*', linestyle='--', label=label_b, color=color_b)

    # Regressionslinien hinzufügen
    if show_regression:
        mask_a = df_sorted[col_tardiness_a].notna()
        x_a = x[mask_a]
        y_a = df_sorted.loc[mask_a, col_tardiness_a]

        mask_b = df_sorted[col_tardiness_b].notna()
        x_b = x[mask_b]
        y_b = df_sorted.loc[mask_b, col_tardiness_b]

        model_a = LinearRegression().fit(x_a, y_a)
        model_b = LinearRegression().fit(x_b, y_b)

        plt.plot(x_a, model_a.predict(x_a), linestyle='-', color=color_a_dark, label=f'{label_a} Trend')
        plt.plot(x_b, model_b.predict(x_b), linestyle='-', color=color_b_dark, label=f'{label_b} Trend')

    # Achsen und Layout
    plt.xlabel(f'{col_job} (sortiert nach {col_arrival})')
    plt.ylabel('Tardiness')
    plt.title(f'Tardiness-Vergleich ({label_a} vs. {label_b})')
    plt.xticks(ticks=x.flatten(), labels=jobs, rotation=rotation)
    plt.yticks(y_ticks)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return df_sorted

def plot_mean_and_max_tardiness_earliness(df_plan_last_ops_list,
                                          title="Tardiness und Earliness pro Tag",
                                          ylabel="Abweichung der Zeit (in Minuten)",
                                          subtitle=None,
                                          show_max=True):
    mean_tardiness_per_day = []
    mean_earliness_per_day = []
    max_tardiness_per_day = []
    max_earliness_per_day = []

    for df in df_plan_last_ops_list:
        mean_tardiness_per_day.append(df["Tardiness"].mean())
        mean_earliness_per_day.append(df["Earliness"].mean())
        if show_max:
            max_tardiness_per_day.append(df["Tardiness"].max())
            max_earliness_per_day.append(df["Earliness"].max())

    days = list(range(len(df_plan_last_ops_list)))

    plt.figure(figsize=(10, 6))

    # Mittelwerte
    plt.plot(days, mean_tardiness_per_day, marker='o', label='Ø Tardiness')
    plt.plot(days, mean_earliness_per_day, marker='s', label='Ø Earliness')

    # Max-Werte nur wenn gewünscht
    if show_max:
        plt.plot(days, max_tardiness_per_day, marker='^', linestyle='--', label='Max Tardiness')
        plt.plot(days, max_earliness_per_day, marker='v', linestyle='--', label='Max Earliness')

    plt.xlabel("Zeit (in Tagen)")
    plt.ylabel(ylabel)

    if subtitle:
        title = f"{title} – {subtitle}"
    plt.title(title)
    
    # Legende oben rechts im Plotbereich
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()