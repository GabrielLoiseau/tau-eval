
from tau_eval.metrics.rouge import compute_rouge

def test_compute_rouge():
    input_texts = ["hello world"]
    output_texts = ["hello world"]
    result = compute_rouge(input_texts, output_texts)
    assert isinstance(result, dict)
    assert "rouge1" in result
    assert "rouge2" in result
    assert "rougeL" in result
    assert result["rouge1"][0] == 1.0
