import copy
from typing import Callable

import numpy as np
import tasknet as tn
from datasets import Dataset

from tau_eval.metrics.bertscore import compute_bertscore
from tau_eval.metrics.cola import compute_cola, load_cola
from tau_eval.metrics.luar import compute_luar, load_luar
from tau_eval.metrics.meteor import compute_meteor
from tau_eval.metrics.nli import compute_nli, load_nli
from tau_eval.metrics.perplexity import compute_perplexity
from tau_eval.metrics.rouge import compute_rouge
from tau_eval.metrics.sbert import compute_sbert, load_sbert
from tau_eval.tasks.customtask import CustomTask

from .logger import logger


# Type for metric functions
MetricFunction = Callable[[str | list[str], str | list[str]], dict[str, float]]

VALID_METRICS = [
    "bertscore",
    "cola",
    "luar",
    "meteor",
    "nli",
    "perplexity",
    "rouge",
    "sbert",
]
DEFAULT_METRICS = ["rouge", "meteor", "luar"]


def evaluate_system_output(
    inputs: list[str],
    outputs: list[str],
    metrics: list[str | MetricFunction] = DEFAULT_METRICS,
) -> dict:
    r"""
    Evaluate a system output with automatic metrics
    """
    for metric in metrics:
        if isinstance(metric, str):
            assert metric in VALID_METRICS, f'"{metric}" is not a valid metric. Choose among: {VALID_METRICS}'
        elif not callable(metric):
            raise TypeError(f"Metric must be a string or callable, got {type(metric)}")

    # Compute each metric
    metric_scores = {}
    if "bertscore" in metrics:
        scores = compute_bertscore(inputs, outputs)
        metric_scores["bertscore_precision"] = np.mean(scores["precision"])
        metric_scores["bertscore_recall"] = np.mean(scores["recall"])
        metric_scores["bertscore_f1"] = np.mean(scores["f1"])

    if "cola" in metrics:
        cola_tokenizer, cola_model = load_cola()
        scores = compute_cola(outputs, cola_tokenizer, cola_model)
        metric_scores["cola"] = np.mean(scores["cola"])

    if "luar" in metrics:
        sim_model = load_luar()
        scores = compute_luar(inputs, outputs, sim_model)
        metric_scores["luar"] = np.mean(scores["luar"])

    if "meteor" in metrics:
        scores = compute_meteor(inputs, outputs)
        metric_scores["meteor"] = np.mean(scores["meteor"])

    if "nli" in metrics:
        nli_tokenizer, nli_model = load_nli()
        scores = compute_nli(inputs, outputs, nli_tokenizer, nli_model)
        metric_scores["nli"] = np.mean(scores["entailment"])

    if "perplexity" in metrics:
        scores = compute_perplexity(outputs)
        metric_scores["perplexity"] = np.mean(scores["perplexities"])

    if "rouge" in metrics:
        scores = compute_rouge(inputs, outputs)
        metric_scores["rouge1"] = np.mean(scores["rouge1"])
        metric_scores["rouge2"] = np.mean(scores["rouge2"])
        metric_scores["rougeL"] = np.mean(scores["rougeL"])

    if "sbert" in metrics:
        sim_model = load_sbert()
        scores = compute_sbert(inputs, outputs, sim_model)
        metric_scores["sbert"] = np.mean(scores["sbert"])

    # Handle custom metric functions
    for func in list(filter(lambda x: callable(x), metrics)):
        custom_scores = func(inputs, outputs)
        metric_scores.update(custom_scores)

    return metric_scores


def run_models_on_task(
    models,
    task,
    metrics,
    classifier_name="answerdotai/ModernBERT-base",
    do_train=False,
    do_train_adversarial=False,
    device="cuda",
    export_generated_texts=True,
):
    ori_task = copy.deepcopy(task)
    args = {"model_name": classifier_name, "evaluation_strategy": None}
    results = {"metrics": metrics}

    m = tn.Model([ori_task], args)
    trainer = tn.Trainer(m, [ori_task], args)
    if do_train:
        trainer.train()

    ori_summary = trainer.evaluate(metric_key_prefix="test")
    results["original_metrics"] = ori_summary[0]

    if "cola" in metrics:
        cola_tokenizer, cola_model = load_cola()
        scores_input = compute_cola(ori_task.dataset["test"][ori_task.s1], cola_tokenizer, cola_model)
        results["original_metrics"]["cola"] = np.mean(scores_input["cola"])

    if export_generated_texts:
        tests = {"original": ori_task.dataset["test"][ori_task.s1]}

    for i, model in enumerate(models):
        logger.debug(f"Evaluating model {i}")
        new_task = copy.deepcopy(task)
        if hasattr(model, "anonymize_batch"):

            def rewrite_batch(batch):
                batch[new_task.s1] = model.anonymize_batch(batch[new_task.s1])
                if new_task.s2 != "" and new_task.s2 in batch.keys():
                    batch[new_task.s2] = model.anonymize_batch(batch[new_task.s2])
                return batch

            if do_train_adversarial:
                new_task.dataset["train"] = new_task.dataset["train"].map(
                    rewrite_batch,
                    batched=True,
                    batch_size=64,
                    features=new_task.dataset["train"].features,
                )
                new_task.dataset["validation"] = new_task.dataset["validation"].map(
                    rewrite_batch,
                    batched=True,
                    batch_size=64,
                    features=new_task.dataset["validation"].features,
                )

            new_task.dataset["test"] = new_task.dataset["test"].map(
                rewrite_batch,
                batched=True,
                batch_size=64,
                features=new_task.dataset["test"].features,
            )
        else:

            def rewrite(example):
                example[new_task.s1] = model.anonymize(example[new_task.s1])
                # example[new_task.s1] = "...."
                if new_task.s2 != "" and new_task.s2 in example.keys():
                    example[new_task.s2] = model.anonymize(example[new_task.s2])
                return example

            if do_train_adversarial:
                new_task.dataset["train"] = new_task.dataset["train"].map(
                    rewrite_batch,
                    batched=True,
                    batch_size=64,
                    features=new_task.dataset["train"].features,
                )
                new_task.dataset["validation"] = new_task.dataset["validation"].map(
                    rewrite_batch,
                    batched=True,
                    batch_size=64,
                    features=new_task.dataset["validation"].features,
                )

            new_task.dataset["test"] = new_task.dataset["test"].map(
                rewrite, features=new_task.dataset["test"].features
            )

        metric_scores = evaluate_system_output(
            ori_task.dataset["test"][ori_task.s1],
            new_task.dataset["test"][new_task.s1],
            metrics,
        )

        results["metrics"] = list(metric_scores.keys())

        new_trainer = tn.Trainer(trainer.model, [new_task], args)
        trainer.test_dataset = new_trainer.test_dataset
        if do_train_adversarial:
            trainer.train_dataset = new_trainer.train_dataset
            trainer.eval_dataset = new_trainer.eval_dataset
        new_summary = trainer.evaluate(metric_key_prefix="test")

        metric_scores = metric_scores | new_summary[0]
        if hasattr(model, "name") and hasattr(task, "name"):
            results[f"{model.name}"] = metric_scores
            if export_generated_texts:
                tests[f"{model.name}"] = new_task.dataset["test"][new_task.s1]
        else:
            results[f"{model.__class__.__name__}"] = metric_scores
            if export_generated_texts:
                tests[f"{model.__class__.__name__}"] = new_task.dataset["test"][new_task.s1]
    if export_generated_texts:
        ds = Dataset.from_dict(tests)
        try:
            ds.save_to_disk(f"{task.name}.hf")
        except Exception:
            ds.save_to_disk(f"{task.__class__.__name__}.hf")
        logger.debug("Saved generated dataset")

    return results


def run_models_on_custom_task(models, task: CustomTask, metrics):
    ori_task = copy.deepcopy(task)
    results = {"metrics": metrics}

    ori_summary = ori_task.evaluate(ori_task.dataset[ori_task.s1])
    results["original_metrics"] = ori_summary

    if "cola" in metrics:
        cola_model, cola_tokenizer = load_cola()
        scores_input = compute_cola(ori_task.dataset[ori_task.s1], cola_tokenizer, cola_model)
        results["original_metrics"]["cola"] = np.mean(scores_input["cola"])

    for i, model in enumerate(models):
        logger.debug(f"Evaluating model {i}")
        new_task = copy.deepcopy(task)
        if hasattr(model, "anonymize_batch"):

            def rewrite_batch(batch):
                batch[new_task.s1] = model.anonymize_batch(batch[new_task.s1])
                return batch

            new_task.dataset = new_task.dataset.map(
                rewrite_batch,
                batched=True,
                batch_size=64,
                features=new_task.dataset.features,
            )
        else:

            def rewrite(example):
                example[new_task.s1] = model.anonymize(example[new_task.s1])
                # example[new_task.s1] = "...."
                return example

            new_task.dataset = new_task.dataset.map(rewrite, features=new_task.dataset.features)

        metric_scores = evaluate_system_output(ori_task.dataset[ori_task.s1], new_task.dataset[new_task.s1], metrics)

        results["metrics"] = list(metric_scores.keys())

        new_summary = ori_task.evaluate(new_task.dataset[new_task.s1])

        metric_scores = metric_scores | new_summary
        results["metrics"] += list(new_summary.keys())
        if hasattr(model, "name") and hasattr(task, "name"):
            results[f"{model.name}"] = metric_scores
        else:
            results[f"{model.__class__.__name__}"] = metric_scores

    return results
