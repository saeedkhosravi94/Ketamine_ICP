import pandas as pd

from experiment_runner import ExperimentRunner
from experiment_configs import EXPERIMENTS
from utils import save_results_to_csv


runner = ExperimentRunner(
    data_path="../data/Ketamine_icp.csv",
    device_type="GPU"
)

all_rows = []

for experiment_name in EXPERIMENTS.keys():
    print("=" * 100)
    print(f"Running experiment: {experiment_name}")
    print("=" * 100)

    result = runner.run(experiment_name)

    for row in result["results_table"]:
        row_copy = row.copy()
        row_copy["Experiment"] = experiment_name
        all_rows.append(row_copy)

all_df = pd.DataFrame(all_rows).sort_values(
    by=["Test_AUC", "Test_Sensitivity", "Test_Specificity"],
    ascending=[False, False, False]
)

all_df.to_csv("all_experiments_summary.csv", index=False)
print(all_df.to_string(index=False))
print("Saved all experiments summary to all_experiments_summary.csv")