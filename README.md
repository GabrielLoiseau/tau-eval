# 𝜏 Tau-Eval: A Unified Evaluation Framework for Useful and Private Text Anonymization

**Tau-Eval** is a user-friendly, modular, and customizable Python library designed to benchmark and evaluate text anonymization algorithms. It enables granular analysis of anonymization impacts from both privacy and utility perspectives. Tau-Eval seamlessly integrates with [LiteLLM](https://www.litellm.ai/) and [🤗 Hugging Face](https://huggingface.co/) to support a wide range of datasets, models, and evaluation metrics.

<div align="center">

[![GNU-GPLv3](https://img.shields.io/badge/license-%20%20GNU%20GPLv3%20-green?style=plastic)](LICENSE)
[![v0.1.0](https://img.shields.io/badge/pypi-v0.1.0-orange)](https://pypi.org/project/tau-eval/0.2.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/downloads/)
[![Tutorials](https://img.shields.io/badge/tutorials-colab-orange)](https://github.com/gabrielloiseau/tau-eval/tree/main/examples)

</div>


## Installation

### From PyPI

Install Tau-Eval via pip:

```
pip install tau-eval
```

### From source

To install from source:

1) Clone this repository on your own path:
```
git clone https://github.com/gabrielloiseau/tau-eval.git
cd tau-eval
```

2) Create an environment with your own preferred package manager. We used [python 3.10](https://www.python.org/downloads/release/python-3100/) and dependencies listed in [`pyproject.toml`](pyproject.toml). If you use [conda](https://docs.conda.io/en/latest/), you can just run the following commands from the root of the project:

```
conda create --name taueval python=3.10         # create the environment
conda activate taueval                         # activate the environment
pip install --user -r pyproject.toml           # install the required packages
```


## Quickstart

Tau-Eval is designed for flexibility. With just a few lines of code, you can set up and run evaluations.

#### 1. Define Your Anonymization Model

Create a custom anonymization model by extending the Anonymizer interface:
```python
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

```
Or use prebuilt models from `tau_eval.models`.

#### 2. Configure Evaluation Metrics
Use built-in metrics from `tau_eval.metrics` or define your own following this signature:
```python
Callable[[str | list[str], str | list[str]], dict]
```
This allows complete control over what and how you evaluate.

#### 3. Instantiate Tasks
Tasks can be created using prebuilt options in `tau_eval.tasks`, or customized using `CustomTask`. Tau-Eval also supports [tasksource](https://github.com/sileod/tasksource) for dataset integration.
```python
from tau_eval.tasks import DeIdentification
from tasknet import AutoTask

anli = AutoTask("anli/a1")
deid = DeIdentification(dataset="ai4privacy/pii-masking-400k")
```

#### 4. Configure and Run Your Experiment
Define an experiment configuration:
```python
from tau_eval.config import ExperimentConfig

config = ExperimentConfig(
    exp_name="test-experiment",
    classifier_name="answerdotai/ModernBERT-base",
    train_task_models=True,
    train_with_generations=False,
)
```
Run the experiment:
```python
from tau_eval.experiment import Experiment

Experiment(
    models=[TestModel(), ...],
    metrics=["bertscore", "rouge"],
    tasks=[anli, deid],
    config=config
).run()
```
#### 5. Visualize Results

Tau-Eval includes built-in visualization tools to compare model anonymization strategies and evaluation results. You can find them with `tau_eval.visualization`. 


## Tutorials

You can explore our tutorials to master **Tau-Eval** more effectively in the [`examples/`](https://github.com/gabrielloiseau/tau-eval/tree/main/examples) folder.


## Contributors

- **[Gabriel Loiseau](https://gabrielloiseau.github.io)**, *Hornetsecurty, Inria Lille*


## Citation

If you use 𝜏 **Tau-Eval** in your work, please cite our paper as follows:

```
@misc{loiseau2025taueval,
      title={Tau-Eval: A Unified Evaluation Framework for Useful and Private Text Anonymization}, 
      author={Gabriel Loiseau, Damien Sileo, Damien Riquet, Maxime Meyer, Marc Tommasi},
      year={2025},
      eprint={2506.05979},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.05979}, 
}
```