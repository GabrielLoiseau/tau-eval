Quickstart
==========

Tau-Eval is designed for flexibility. With just a few lines of code, you can set up and run evaluations.

1. Define Your Anonymization Model
-----------------------------------

Create a custom anonymization model by extending the Anonymizer interface:

.. code-block:: python

    from tau_eval.models import Anonymizer

    class TestModel(Anonymizer):
        def __init__(self):
            self.name = "Test Model"

        def anonymize(self, text: str) -> str:
            # Implement anonymization logic
            return text

        def anonymize_batch(self, texts: list[str]) -> list[str]:
            # Batch processing
            return texts

Or use prebuilt models from `tau_eval.models`.

2. Configure Evaluation Metrics
-------------------------------

Use built-in metrics from `tau_eval.metrics` or define your own following this signature:

.. code-block:: python

    Callable[[str | list[str], str | list[str]], dict]

This allows complete control over what and how you evaluate.

3. Instantiate Tasks
--------------------

Tasks can be created using prebuilt options in `tau_eval.tasks`, or customized using `CustomTask`. Tau-Eval also supports [tasksource](https://github.com/sileod/tasksource) for dataset integration.

.. code-block:: python

    from tau_eval.tasks import DeIdentification
    from tasknet import AutoTask

    anli = AutoTask("anli/a1")
    deid = DeIdentification(dataset="ai4privacy/pii-masking-400k")

4. Configure and Run Your Experiment
------------------------------------

Define an experiment configuration:

.. code-block:: python

    from tau_eval.config import ExperimentConfig

    config = ExperimentConfig(
        exp_name="test-experiment",
        classifier_name="answerdotai/ModernBERT-base",
        train_task_models=True,
        train_with_generations=False,
    )

Run the experiment:

.. code-block:: python

    from tau_eval.experiment import Experiment

    Experiment(
        models=[TestModel(), ...],
        metrics=["bertscore", "rouge"],
        tasks=[anli, deid],
        config=config
    ).run()

5. Visualize Results
--------------------

Tau-Eval includes built-in visualization tools to compare model anonymization strategies and evaluation results. You can find them with `tau_eval.visualization`.
