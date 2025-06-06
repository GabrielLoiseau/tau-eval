import json
import os
from dataclasses import dataclass, field

import pandas as pd
from rich.table import Table
from tasknet import Task

from tau_eval.tasks.customtask import CustomTask

from .logger import logger
from .models import Anonymizer
from .utils import run_models_on_custom_task, run_models_on_task


def rich_display_dataframe(df, title="Dataframe") -> None:
    """Display dataframe as table using rich library.
    Args:
        df (pd.DataFrame): dataframe to display
        title (str, optional): title of the table. Defaults to "Dataframe".
    Raises:
        NotRenderableError: if dataframe cannot be rendered
    Returns:
        rich.table.Table: rich table
    """
    from rich import print

    # ensure dataframe contains only string values
    df = df.astype(str)

    table = Table(title=title)
    for col in df.columns:
        table.add_column(col)
    for row in df.values:
        table.add_row(*row)
    print(table)

@dataclass
class ExperimentConfig:
    """Evaluation experiment config"""

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    classifier_name: str = "answerdotai/ModernBERT-base"
    train_task_models: bool = False
    train_with_generations: bool = False
    device: str | None = "cuda"
    classifier_args: dict = field(default_factory=dict)


class Experiment:
    def __init__(
        self,
        models: list[Anonymizer],
        metrics: list[str],
        tasks: list[Task],
        config: ExperimentConfig = ExperimentConfig(),
    ):
        self.models = models
        self.metrics = metrics
        self.tasks = tasks
        self.args = config

    def run(self, output_dir="", device="cuda"):
        self.output_dir = output_dir
        logger.info("Running experiment...")
        out = {}

        for i, task in enumerate(self.tasks):
            logger.info(f"Running task: {i}")
            if isinstance(task, CustomTask):
                results = run_models_on_custom_task(self.models, task, self.metrics)
            else:
                results = run_models_on_task(
                    self.models,
                    task,
                    self.metrics,
                    self.args.classifier_name,
                    self.args.train_task_models,
                    self.args.train_with_generations,
                    device,
                )
            if hasattr(task, "name"):
                out[f"{task.name}_{i}"] = results
            else:
                out[f"{task.__class__.__name__}_{i}"] = results

        self.results = out
        with open(self.output_dir, "w") as f:
            json.dump(out, f)
            logger.info("Results saved")

    def summary(self, output_dir=None, to_rich=False):
        if output_dir is not None:
            results = json.load(open(output_dir))
        else:
            results = json.load(open(self.output_dir))

        task_dataframes = {}

        for task_name, task in results.items():
            # Prepare columns for DataFrame
            if isinstance(task, CustomTask):
                columns = ["Model Name"] + task["metrics"]
                rows = []

                # Add original metrics
                original_metrics = task.get("original_metrics", {})
                if original_metrics:
                    original_row = ["Original"]
                    for m in task["metrics"]:
                        if m == "cola":
                            original_row.append(round(original_metrics.get("cola", 0), 4))
                        else:
                            original_row.append("-")
                    rows.append(original_row)
                    # Add rewritten metrics
                for model_name, model in task.items():
                    if model_name in ["original_metrics", "metrics"]:
                        continue

                    row = [model_name]

                    for m in task["metrics"]:
                        row.append(round(model.get(m, 0), 4))

                    rows.append(row)

            else:
                columns = ["Model Name", "Accuracy", "F1"] + task["metrics"]
                rows = []

                # Add original metrics
                original_metrics = task.get("original_metrics", {})
                if original_metrics:
                    original_row = ["Original"]
                    original_row.append(round(original_metrics.get("test_accuracy", 0), 4))
                    original_row.append(round(original_metrics.get("test_f1", 0), 4))
                    for m in task["metrics"]:
                        if m == "cola":
                            original_row.append(round(original_metrics.get("cola", 0), 4))
                        else:
                            original_row.append("-")
                    rows.append(original_row)

                # Add rewritten metrics
                for model_name, model in task.items():
                    if model_name in ["original_metrics", "metrics"]:
                        continue

                    row = [model_name]
                    row.append(model.get("test_accuracy", 0))
                    row.append(model.get("test_f1", 0))

                    for m in task["metrics"]:
                        row.append(round(model.get(m, 0), 4))

                    rows.append(row)

            # Create DataFrame for the task
            task_dataframe = pd.DataFrame(rows, columns=columns)
            task_dataframes[task_name] = task_dataframe
        if to_rich:
            for t, value in task_dataframes.items():
                rich_display_dataframe(value)

        return task_dataframes

    def plot_tradeoff(
        self,
        task_name1,
        target1,
        task_name2,
        target2,
        x_label="Metric 1",
        y_label="Metric 2",
        title="Tradeoff Between Metrics",
        figsize=(4, 4),
    ):
        from itertools import cycle

        import matplotlib.pyplot as plt

        task1 = self.results[task_name1]
        task2 = self.results[task_name2]
        model_names = []
        x_vals = []
        y_vals = []

        for model_name, model in task1.items():
            model_names.append(model_name)
            if model_name in ["original_metrics", "metrics"]:
                continue
            else:
                x_vals.append(model.get(target1))

        for model_name, model in task2.items():
            model_names.append(model_name)
            if model_name in ["original_metrics", "metrics"]:
                continue
            else:
                y_vals.append(model.get(target2))

        markers = cycle(
            ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H"]
        )  # Cycle through different marker styles
        plt.figure(figsize=figsize)

        for i, (x_val, y_val) in enumerate(zip(x_vals, y_vals)):
            marker = next(markers)
            plt.scatter(
                x_val,
                y_val,
                label=model_names[i] if model_names else f"Model {i + 1}",
                marker=marker,
                s=100,
                alpha=0.7,
                edgecolor="k",
            )
            # Annotate points with model names if provided
            if model_names:
                plt.text(x_val, y_val, model_names[i], fontsize=9, ha="right", va="bottom")

        plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        plt.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        plt.grid(alpha=0.3)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend(loc="best", fontsize=9)
        plt.tight_layout()
