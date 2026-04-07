import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from experiment_runner import ExperimentRunner
from utils import plot_dolan_more, plot_roc_curves, save_results_to_csv

runner = ExperimentRunner(
    data_path="../data/Ketamine_icp.csv",
    device_type="GPU"
)

result = runner.run("impute_5_cnn_bilstm_attention_SASMOTE_k100_300")

plot_roc_curves(
    result["roc_entries"],
    title=f'ROC Curve Comparison - {result["experiment_name"]}'
)

save_results_to_csv(
    result["results_table"],
    filepath=f'{result["experiment_name"]}_results.csv'
)

plot_dolan_more(result["perf_profile_data"], title=f'Performance Profile - {result["experiment_name"]}')


