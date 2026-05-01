from aoi_sentinel.eval.metrics import score


def test_perfect():
    s = score([1, 0, 1, 0], [1, 0, 1, 0])
    assert s.true_defect_recall == 1.0
    assert s.false_call_reduction == 1.0
    assert s.escape_rate == 0.0
    assert s.accuracy == 1.0


def test_one_escape():
    s = score([1, 1, 0, 0], [0, 1, 0, 0])
    assert s.true_defect_recall == 0.5
    assert s.escape_rate == 0.5
    assert s.false_call_reduction == 1.0
