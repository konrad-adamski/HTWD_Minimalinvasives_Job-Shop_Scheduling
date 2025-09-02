import json
import matplotlib.pyplot as plt
from config.project_config import get_data_path

# ------------------------------------------------------
# 1) JSON-Datei laden
# ------------------------------------------------------
basic_data_path = get_data_path("basic")
file_path = basic_data_path / "jobshop_instances.json"

with open(file_path, "r", encoding="utf-8") as f:
    jobshop_instances = json.load(f)

instance = jobshop_instances["instance ft10"]

# ------------------------------------------------------
# 2) Zeilen vorbereiten (Dauer zweistellig!)
# ------------------------------------------------------
lines = []
for routing_id, operations in instance.items():
    op_tuples = [(op["machine"], op["duration"]) for op in operations]
    # Maschine normal, Dauer zweistellig
    op_str = "  ".join(f"({m}, {d:02d})" for m, d in op_tuples)
    lines.append(f"Job {routing_id}:  {op_str}")

# ------------------------------------------------------
# 3) Dynamische Figure-Größe berechnen
# ------------------------------------------------------
max_len = max(len(line) for line in lines)
fig_w = max(6, 0.095 * max_len)  # Faktor 0.095 passt ganz gut
fig_h = 0.6 + 0.3 * len(lines)

fig = plt.figure(figsize=(fig_w, fig_h))
ax = plt.gca()
ax.axis("off")

# ------------------------------------------------------
# 4) Textzeilen plotten
# ------------------------------------------------------
line_spacing = 1  # < 1 = enger, > 1 = weiter
for i, line in enumerate(lines):
    y = 1 - (i + 1) * (line_spacing / (len(lines) + 1))
    ax.text(
        0.01, y, line,
        family="monospace",
        fontsize=14,
        va="top", ha="left",
    )

# ------------------------------------------------------
# 5) Exportieren
# ------------------------------------------------------
plt.savefig("ft10_jobs.pdf", bbox_inches="tight", pad_inches=0.04)
plt.close(fig)

print("Exportiert als jobshop_instance.svg und jobshop_instance.pdf")
