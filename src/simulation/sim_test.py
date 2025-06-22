from src.simulation.ProductionSimulation import ProductionSimulation
from configs.config import get_path
import pandas as pd

basic_data_path = get_path("data", "basic")
df_schedule = pd.read_csv(basic_data_path / "schedule_example.csv")

print("Maschinenbelegungsplan:")
print(df_schedule)
print("\n", "---"*60)


print("Simulation:")
simulation = ProductionSimulation(df_schedule, sigma=0.45)
df_execution = simulation.run(start_time = 0, end_time=None)
print(df_execution)
print("\n", "---"*60)

# Schritt 1: Relevante Spalten aus df_execution
executed_keys = df_execution[[ "Job", "Operation" ]].drop_duplicates()

# Schritt 2: Negative Merge – finde alle, die NICHT ausgeführt wurden
df_plan_undone = df_schedule.merge(
    executed_keys,
    on=["Job", "Operation"],
    how="left",
    indicator=True
)

print("Unerledigte Operationen")
# Schritt 3: Nur die Zeilen behalten, die NICHT in df_execution waren
df_plan_undone = df_plan_undone[df_plan_undone["_merge"] == "left_only"].drop(columns="_merge").reset_index(drop=True)
print(df_plan_undone)



