import os
import sys
import copy
import pandas as pd

sys.path.append("..")

from experiment_runner import ExperimentRunner
import experiment_configs
from bilstm_configs import BILSTM_EXPERIMENTS

runner = ExperimentRunner(
    data_path="../../data/Ketamine_icp.csv",
    device_type="GPU"
)

all_rows = []

for experiment_name, cfg in BILSTM_EXPERIMENTS.items():
    print("=" * 100)
    print(f"Running experiment: {experiment_name}")
    print("=" * 100)

    old_value = experiment_configs.EXPERIMENTS.get(experiment_name)
    experiment_configs.EXPERIMENTS[experiment_name] = copy.deepcopy(cfg)

    result = runner.run(experiment_name)

    for row in result["results_table"]:
        row_copy = row.copy()
        row_copy["Experiment"] = experiment_name
        all_rows.append(row_copy)

    if old_value is None:
        del experiment_configs.EXPERIMENTS[experiment_name]
    else:
        experiment_configs.EXPERIMENTS[experiment_name] = old_value

os.makedirs("results", exist_ok=True)

all_df = pd.DataFrame(all_rows).sort_values(
    by=["Test_AUC", "Test_Sensitivity", "Test_Specificity"],
    ascending=[False, False, False]
)

out_path = "results/bilstm_experiments_summary.csv"
all_df.to_csv(out_path, index=False)

print(all_df.to_string(index=False))
print(f"Saved results to {out_path}")