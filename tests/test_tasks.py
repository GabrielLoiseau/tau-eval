
import pytest
from tau_eval.tasks.customtask import CustomTask

def test_custom_task_evaluate():
    with pytest.raises(NotImplementedError):
        CustomTask().evaluate([])
