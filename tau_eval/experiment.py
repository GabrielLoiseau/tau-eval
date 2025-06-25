import json
import os
from dataclasses import dataclass, field
from typing import Callable

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
        metrics: list[str | Callable],
        tasks: list[Task | CustomTask],
        config: ExperimentConfig = ExperimentConfig(),
    ):
        self.models = models
        self.metrics = metrics
        self.tasks = tasks
        self.args = config
        self.results = None
        self.output_dir = None

    def run(self, output_dir="results.json"):
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
                    self.args.device,
                )
            if hasattr(task, "name"):
                out[f"{task.name}_{i}"] = results
            else:
                out[f"{task.__class__.__name__}_{i}"] = results

        self.results = out
        with open(self.output_dir, "w") as f:
            json.dump(out, f)
            logger.info("Results saved")


    @classmethod
    def from_json(cls, filepath: str):
        """Loads experiment results from a JSON file."""
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"The specified JSON file was not found: {filepath}")

        with open(filepath, "r") as f:
            try:
                loaded_data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {filepath}: {e}")
                raise ValueError(f"Invalid JSON format in {filepath}: {e}")

        # Create a new experiment instance with empty lists for models, metrics, tasks
        exp = cls(models=[], metrics=[], tasks=[], config=ExperimentConfig())
        exp.results = loaded_data
        exp.output_dir = filepath

        logger.info(f"Experiment results loaded successfully from {filepath}")
        return exp

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

