import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr, spearmanr
import numpy as np

def get_all_dataset_names(data):
    """Extracts all dataset names from the Experiment data."""
    return list(data.keys())


def get_all_model_method_names(data):
    """Extract unique model names across all datasets"""
    models = set()
    for dataset_data in data.values():
        for key in dataset_data.keys():
            if key not in ["metrics", "original_metrics"]:
                models.add(key)
    return sorted(models)


def get_all_numeric_metric_names(data):
    """
    Extracts all unique numeric metric names present in model/method results
    or in 'original_metrics'.
    """
    all_metrics = set()
    utility_keys_to_ignore = [
        "test_name",
        "test_size",
        "test_index",
        "test_runtime",
        "test_samples_per_second",
        "test_steps_per_second",
        "epoch",
        "Accuracy",  # Accuracy is often NaN
    ]
    for dataset_name, dataset_data in data.items():
        # Check original_metrics
        if "original_metrics" in dataset_data and isinstance(
            dataset_data["original_metrics"], dict
        ):
            for metric_key, metric_value in dataset_data["original_metrics"].items():
                if metric_key not in utility_keys_to_ignore and isinstance(
                    metric_value, (int, float)
                ):
                    all_metrics.add(metric_key)

        # Check each model/method
        for model_or_method_key, model_or_method_data in dataset_data.items():
            if model_or_method_key not in [
                "metrics",
                "original_metrics",
            ] and isinstance(model_or_method_data, dict):
                for metric_key, metric_value in model_or_method_data.items():
                    if metric_key not in utility_keys_to_ignore and isinstance(
                        metric_value, (int, float)
                    ):
                        all_metrics.add(metric_key)
    return sorted(all_metrics)


def _shorten_dataset_name(d_name):
    """Helper to shorten dataset names for display."""
    return d_name.split("/")[0].replace("___", "\n").replace("_", " ").title()[:30]


def _shorten_model_name(m_name, max_len=25):
    """Helper to shorten model names for display."""
    if len(m_name) > max_len:
        return m_name[: max_len - 3] + "..."
    return m_name

# Visualization Functions

def plot_metric_comparison_across_datasets(
    data, metric_name, specific_models=None, show_original=True
):
    """
    Compares a specific metric for selected models across all datasets.

    Args:
        data: The parsed JSON data from Experiment.
        metric_name: The metric to compare (e.g., 'bertscore_f1', 'test_accuracy').
        specific_models (optional): A list of model names to include.
                                          If None, includes all found models.
        show_original: If True, includes 'Original Model' performance for the metric.
    """
    datasets = get_all_dataset_names(data)
    all_model_methods = get_all_model_method_names(data)

    if specific_models:
        models_to_plot = [m for m in all_model_methods if m in specific_models]
        if show_original and "Original Model" not in models_to_plot:
            models_to_plot.append("Original Model")
    else:
        models_to_plot = [
            m for m in all_model_methods if m != "metrics" and m != "original_metrics"
        ]
        if not show_original and "Original Model" in models_to_plot:
            models_to_plot.remove("Original Model")

    if not models_to_plot:
        print("No models/methods selected or found for plotting.")
        return

    metric_values = {}  # {model_name: [val_ds1, val_ds2, ...]}

    for model_name in models_to_plot:
        metric_values[model_name] = []
        for dataset_name in datasets:
            dataset_info = data.get(dataset_name, {})
            score = np.nan
            if model_name == "Original Model":
                model_info = dataset_info.get("original_metrics", {})
                val = model_info.get(metric_name)
                if isinstance(val, (int, float)):
                    score = val
            else:
                model_info = dataset_info.get(model_name, {})
                val = model_info.get(metric_name)
                if isinstance(val, (int, float)):
                    score = val
            metric_values[model_name].append(score)

    # Filter out models that have no scores at all for this metric
    models_with_data = [
        m for m in models_to_plot if not all(np.isnan(s) for s in metric_values[m])
    ]
    if not models_with_data:
        print(
            f"No data found for metric '{metric_name}' for the selected models across datasets."
        )
        return
    models_to_plot = models_with_data

    num_datasets = len(datasets)
    num_models_plotted = len(models_to_plot)

    fig, ax = plt.subplots(
        figsize=(max(15, num_datasets * 1.5 * (num_models_plotted / 5)), 8)
    )

    index = np.arange(num_datasets)
    # Adjust bar width based on number of models to avoid excessive crowding/sparseness
    total_group_width = 0.8
    bar_width = total_group_width / num_models_plotted

    for i, model_name in enumerate(models_to_plot):
        scores = np.array(metric_values[model_name], dtype=float)
        ax.bar(
            index + i * bar_width - (total_group_width - bar_width) / 2,
            scores,
            bar_width,
            label=_shorten_model_name(model_name),
        )

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{metric_name} by Model Across Datasets", fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels(
        [_shorten_dataset_name(d) for d in datasets],
        rotation=45,
        ha="right",
        fontsize=10,
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout for legend
    plt.show()


def plot_all_metrics_for_model_on_dataset(data, dataset_name, model_name):
    """
    Plots all numeric metrics for a specific model on a specific dataset.

    Args:
        data: The parsed JSON data from Experiment.
        dataset_name: The name of the dataset.
        model_name: The name of the model (e.g., 'google/gemini-flash-1.5-8b').
                                    Use "Original Model" to see metrics from original texts.
    """
    dataset_info = data.get(dataset_name)
    if not dataset_info:
        print(f"Dataset '{dataset_name}' not found.")
        return

    metrics_source_dict = None
    if model_name == "Original Model":
        metrics_source_dict = dataset_info.get("original_metrics")
        plot_title = f"Original Model Metrics on {_shorten_dataset_name(dataset_name)}"
    else:
        metrics_source_dict = dataset_info.get(model_name)
        plot_title = f"Metrics for {_shorten_model_name(model_name)} on {_shorten_dataset_name(dataset_name)}"

    if not metrics_source_dict or not isinstance(metrics_source_dict, dict):
        print(
            f"Model '{model_name}' not found or has no metric data in dataset '{dataset_name}'."
        )
        return

    utility_keys_to_ignore = [
        "test_name",
        "test_size",
        "test_index",
        "test_runtime",
        "test_samples_per_second",
        "test_steps_per_second",
        "epoch",
        "Accuracy",
    ]

    metrics_data = {
        k: v
        for k, v in metrics_source_dict.items()
        if isinstance(v, (int, float))
        and not np.isnan(v)
        and k not in utility_keys_to_ignore
    }

    if not metrics_data:
        print(
            f"No numeric metrics found for '{model_name}' in dataset '{dataset_name}'."
        )
        return

    sorted_metrics = sorted(
        metrics_data.items(), key=lambda item: item[0]
    )  # Sort by metric name
    metric_names = [item[0] for item in sorted_metrics]
    metric_values = [item[1] for item in sorted_metrics]

    fig, ax = plt.subplots(figsize=(max(10, len(metric_names) * 0.7), 6))
    bars = ax.bar(metric_names, metric_values, color="cornflowerblue")

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(plot_title, fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{yval:.3f}",
            va="bottom",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()


def plot_metric_distribution(data, metric_name, chart_type="hist"):
    """
    Plots the distribution of a specific metric across all models/methods and datasets.

    Args:
        data: The parsed JSON data from Experiment.
        metric_name: The metric whose distribution is to be plotted.
        chart_type: Type of chart: 'hist' for histogram, 'box' for box plot.
    """
    all_scores = []
    sources = []

    for dataset_name, dataset_info in data.items():
        # Original model scores
        original_m_data = dataset_info.get("original_metrics", {})
        score_orig = original_m_data.get(metric_name)
        if isinstance(score_orig, (int, float)) and not np.isnan(score_orig):
            all_scores.append(score_orig)
            sources.append("Original")

        # Other models/methods
        for model_key, model_data_dict in dataset_info.items():
            if model_key not in ["metrics", "original_metrics"] and isinstance(
                model_data_dict, dict
            ):
                score = model_data_dict.get(metric_name)
                if isinstance(score, (int, float)) and not np.isnan(score):
                    all_scores.append(score)
                    sources.append(
                        model_key
                    )

    if not all_scores:
        print(f"No scores found for metric '{metric_name}' to plot distribution.")
        return

    plt.figure(figsize=(10, 6))
    if chart_type == "hist":
        plt.hist(all_scores, bins=20, color="teal", edgecolor="black", alpha=0.7)
        plt.title(f"Distribution of {metric_name} Scores (All Sources)", fontsize=14)
        plt.xlabel(metric_name, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
    elif chart_type == "box":
        plt.boxplot(
            all_scores,
            vert=False,
            patch_artist=True,
            boxprops={"facecolor": "lightblue"},
            medianprops={"color": "red"},
        )
        plt.title(f"Box Plot of {metric_name} Scores (All Sources)", fontsize=14)
        plt.xlabel(metric_name, fontsize=12)
        plt.yticks([])
    else:
        print(f"Unknown chart type: {chart_type}. Use 'hist' or 'box'.")
        return

    plt.grid(True, axis="x" if chart_type == "box" else "y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_radar_model_comparison(
    data,
    metric_name,
    model_list,
    ordered_dataset_keys
):
    """
    Generates a single radar plot to compare specified model series across datasets for a given metric.

    Args:
        data: The parsed JSON data from Experiment.
        metric_name: The metric to plot (e.g., 'sbert', 'test_f1').
        model_list: A list of model/method names to include.
        ordered_dataset_keys: List of dataset names, determining the order of axes on the radar.
    """
    num_vars = len(ordered_dataset_keys)
    if num_vars == 0:
        print("Error: No datasets specified for radar axes.")
        return
    # Prepare angles for radar plot
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Score Extraction
    extracted_scores_for_plot = {}

    for model_name in model_list:
        current_series_scores = []

        for dataset_key in ordered_dataset_keys:
            dataset_level_data = data.get(dataset_key, {})

            model_specific_data = dataset_level_data.get(model_name, {})
            score = model_specific_data.get(metric_name, np.nan)

            if not isinstance(score, (int, float)):
                score = np.nan
            current_series_scores.append(score)

        extracted_scores_for_plot[model_name] = current_series_scores

    # --- Plotting ---
    fig, current_ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar":True})

    for model_name in model_list:
        scores_for_radar = extracted_scores_for_plot.get(model_name, [])
        if not scores_for_radar:
            print(f"Warning: No scores found for series '{model_name}'")
            continue

        plot_values = [float(s) if not isinstance(s, str) else np.nan for s in scores_for_radar]
        plot_values += plot_values[:1] # Close the radar

        current_ax.plot(angles, plot_values, linewidth=2, label=model_name)
        current_ax.fill(angles, plot_values, alpha=0.25)

    current_ax.set_xticks(angles[:-1])
    # Shorten dataset labels if they are full keys
    short_labels = [_shorten_dataset_name(lbl) if '/' in lbl else lbl for lbl in ordered_dataset_keys]
    current_ax.set_xticklabels(short_labels, fontsize=10)

    all_vals_this_plot = []
    for s_label in extracted_scores_for_plot:
        all_vals_this_plot.extend([v for v in extracted_scores_for_plot[s_label] if not np.isnan(v)])
    if all_vals_this_plot:
        min_val_plot = min(all_vals_this_plot)
        max_val_plot = max(all_vals_this_plot)
        padding = (max_val_plot - min_val_plot) * 0.1 if (max_val_plot - min_val_plot) > 0 else 0.1
        current_ax.set_ylim(min_val_plot - padding, max_val_plot + padding if max_val_plot > min_val_plot else max_val_plot + 0.2)


    current_ax.grid(True)
    current_ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=len(model_list), fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()


def plot_trade_off_metric(
    data, x_metric_name, y_metric_name, task_list=None, model_list=None
):
    """
    Creates a scatter plot of a task performance metric vs. an anonymization/utility metric,
    with a legend for different model/method groups.

    Args:
        data: The parsed JSON data from Experiment.
        x_metric_name: e.g., 'test_f1', 'test_accuracy'.
        y_metric_name: e.g., 'bertscore_f1', 'rougeL', 'sbert'.
        model_list (optional): Filter for specific models.
    """
    model_data = {}  # {group_name: {'task': [], 'anon': []}}

    for dataset_name, dataset_info in data.items():
        if task_list and dataset_name not in task_list:
            continue
        for model_key, model_data_dict in dataset_info.items():
            if model_key in ["metrics", "original_metrics"] or model_key not in model_list or not isinstance(
                model_data_dict, dict
            ):
                continue

            task_val = model_data_dict.get(x_metric_name)
            anon_val = model_data_dict.get(y_metric_name)

            if not (
                isinstance(task_val, (int, float))
                and not np.isnan(task_val)
                and isinstance(anon_val, (int, float))
                and not np.isnan(anon_val)
            ):
                continue

            if model_key not in model_data:
                model_data[model_key] = {"task": [], "anon": []}

            model_data[model_key]["task"].append(task_val)
            model_data[model_key]["anon"].append(anon_val)

    if not model_data:
        print("No text found for the selected models and metrics.")
        return

    plt.figure(figsize=(12, 8))

    # Define a list of colors to cycle through for different groups
    color_map = plt.cm.get_cmap("tab20")
    colors = [color_map(i) for i in range(len(model_data))]

    for i, (group_name, scores_dict) in enumerate(model_data.items()):
        if scores_dict["task"]:
            plt.scatter(
                scores_dict["anon"],
                scores_dict["task"],
                alpha=0.7,
                label=group_name,
                color=colors[i % len(colors)],
            )

    plt.xlabel(f"{x_metric_name}", fontsize=12)
    plt.ylabel(f"{y_metric_name}", fontsize=12)

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best", fontsize=10)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    # plt.tight_layout(rect=[0, 0, 0.85, 1]) # if legend is outside

    plt.tight_layout()
    plt.show()


def compute_correlation(data, metric1_name: str, metric2_name: str,
                         method: str = "kendall", aggregate_across_tasks: bool = False, task_name: str = None) -> dict:
    """
    Computes correlation between two metrics for models.

    Args:
        data: Data structure containing metrics for models across tasks
        metric1_name: Name of the first metric
        metric2_name: Name of the second metric
        method: Correlation method ('kendall', 'pearson', 'spearman')
        aggregate_across_tasks: If True, compute single correlation across all tasks
        task_name: Specific task name to compute correlation (ignored if aggregate_across_tasks=True)

    Returns:
        dict of {task_name: (correlation, p_value)} for all tasks
    """

    def _extract_metric_pairs(dataset_info):
        """Extract valid metric pairs from a dataset."""
        pairs = []
        for model_key in get_all_model_method_names(data):
            model_data = dataset_info.get(model_key, {})
            if not isinstance(model_data, dict):
                continue

            val1 = model_data.get(metric1_name)
            val2 = model_data.get(metric2_name)

            if (isinstance(val1, (int, float)) and not np.isnan(val1) and
                isinstance(val2, (int, float)) and not np.isnan(val2)):
                pairs.append((val1, val2))
        return pairs

    def _compute_correlation(pairs):
        """Compute correlation and p-value from metric pairs."""
        if len(pairs) < 2:
            return np.nan, np.nan

        values1, values2 = zip(*pairs)

        # Check for constant values
        if len(set(values1)) < 2 or len(set(values2)) < 2:
            return np.nan, np.nan

        try:
            if method == 'kendall':
                corr, p_val = kendalltau(values1, values2)
            elif method == 'pearson':
                corr, p_val = pearsonr(values1, values2)
            elif method == 'spearman':
                corr, p_val = spearmanr(values1, values2)
            else:
                raise ValueError(f"Unsupported correlation method: {method}")
            return corr, p_val
        except Exception as e:
            print(f"Error computing {method} correlation: {e}")
            return np.nan, np.nan

    # Aggregate across all tasks
    if aggregate_across_tasks:
        if task_name:
            print(f"Warning: task_name '{task_name}' ignored when aggregate_across_tasks=True")

        all_pairs = []
        for dataset_name in get_all_dataset_names(data):
            dataset_info = data.get(dataset_name)
            if dataset_info:
                all_pairs.extend(_extract_metric_pairs(dataset_info))

        return {"aggregated_tasks":_compute_correlation(all_pairs)}

    # Per-task correlation
    tasks_to_process = [task_name] if task_name else get_all_dataset_names(data)

    # Validate single task
    if task_name and task_name not in get_all_dataset_names(data):
        print(f"Warning: Task '{task_name}' not found")
        return {f"{task_name}": (np.nan, np.nan)}

    results = {}
    for current_task in tasks_to_process:
        dataset_info = data.get(current_task)
        if not dataset_info:
            results[current_task] = (np.nan, np.nan)
            continue

        pairs = _extract_metric_pairs(dataset_info)
        results[current_task] = _compute_correlation(pairs)

    return results
