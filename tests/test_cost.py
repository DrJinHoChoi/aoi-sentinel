from aoi_sentinel.sim.cost import (
    ACTION_DEFECT,
    ACTION_ESCALATE,
    ACTION_PASS,
    LABEL_FALSE_CALL,
    LABEL_TRUE_DEFECT,
    CostMatrix,
)


def test_default_matrix_shape():
    m = CostMatrix().matrix()
    assert m.shape == (2, 3)


def test_escape_is_pass_on_true_defect():
    c = CostMatrix()
    assert c.is_escape(LABEL_TRUE_DEFECT, ACTION_PASS)
    assert not c.is_escape(LABEL_TRUE_DEFECT, ACTION_DEFECT)
    assert not c.is_escape(LABEL_FALSE_CALL, ACTION_PASS)


def test_escape_dominates_other_costs():
    c = CostMatrix()
    m = c.matrix()
    escape_cost = m[LABEL_TRUE_DEFECT, ACTION_PASS]
    other_costs = [
        m[LABEL_FALSE_CALL, ACTION_DEFECT],
        m[LABEL_FALSE_CALL, ACTION_ESCALATE],
        m[LABEL_TRUE_DEFECT, ACTION_ESCALATE],
    ]
    for o in other_costs:
        assert escape_cost > 100 * o or o == 0
