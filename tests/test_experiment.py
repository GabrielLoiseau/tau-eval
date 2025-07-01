
import json
import os
from tau_eval.experiment import Experiment, ExperimentConfig
from tau_eval.models.dummy import DummyModel
from tau_eval.tasks.customtask import CustomTask

def test_experiment_run():
    # Create a dummy task
    class DummyTask(CustomTask):
        def __init__(self):
            self.name = "dummy_task"
            self.dataset = {"test": [{"text": "hello world"}]}

        def evaluate(self, new_texts):
            return {"dummy_metric": 1.0}

    # Configure and run the experiment
    models = [DummyModel()]
    metrics = ["rouge"]
    tasks = [DummyTask()]
    config = ExperimentConfig()
    exp = Experiment(models, metrics, tasks, config)
    exp.run("test_results.json")

    # Check that the results file was created
    assert os.path.exists("test_results.json")

    # Check the content of the results file
    with open("test_results.json", "r") as f:
        results = json.load(f)
        assert "dummy_task_0" in results
        task_results = results["dummy_task_0"]
        assert "Dummy Model" in task_results
        assert "rouge1" in task_results["Dummy Model"]

    # Clean up the results file
    os.remove("test_results.json")
