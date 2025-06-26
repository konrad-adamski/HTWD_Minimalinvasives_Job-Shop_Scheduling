# Datenzugriff
from configs.path_manager import get_path
import json

# Utils
import src.utils.presenter as show
from src.utils.production_process import convert
from src.utils.production_process import initialize as init
from src.utils.analize import check_constrains as check

# Solver Model
from src.models.cp import makespan


# Datei laden
basic_data_path = get_path("data", "basic")
file_path = basic_data_path / "jobshop_instances.json"

with open(file_path, "r", encoding="utf-8") as f:
    jobshop_instances = json.load(f)

instance =  jobshop_instances["instance ft10"]
df_instance = convert.jssp_dict_to_df(instance)

df_production_orders = init.production_orders(df_instance)


df_schedule = makespan.solve_jssp(df_production_orders, msg=True, log_file="makespan_cp.log")

print(df_schedule)