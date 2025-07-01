Installation
============

From PyPI
---------

Install Tau-Eval via pip:

.. code-block:: bash

    pip install tau-eval

From source
-----------

To install from source:

1) Clone this repository on your own path:

.. code-block:: bash

    git clone https://github.com/gabrielloiseau/tau-eval.git
    cd tau-eval

2) Create an environment with your own preferred package manager. We used `Python 3.10 <https://www.python.org/downloads/release/python-3100/>`_ and dependencies listed in `pyproject.toml <pyproject.toml>`_. If you use `conda <https://docs.conda.io/en/latest/>`_, you can just run the following commands from the root of the project:

.. code-block:: bash

    conda create --name taueval python=3.10         # create the environment
    conda activate taueval                         # activate the environment
    pip install --user -r pyproject.toml           # install the required packages
