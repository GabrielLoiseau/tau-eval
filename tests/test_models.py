
import pytest
from tau_eval.models.anonymizer import Anonymizer
from tau_eval.models.dummy import DummyModel

def test_dummy_model_anonymize():
    model = DummyModel()
    assert model.anonymize("test") == "..."

def test_dummy_model_anonymize_batch():
    model = DummyModel()
    assert model.anonymize_batch(["test1", "test2"]) == ["...", "..."]

def test_anonymizer_interface():
    with pytest.raises(NotImplementedError):
        Anonymizer().anonymize("test")
