import numpy as np

from models.explain_linear_model import linear_shap_values


class DummyModel:
    coef_ = np.array([[2.0, -1.0]])
    intercept_ = np.array([0.5])


def test_linear_attributions_reconstruct_log_odds():
    background = np.array([[1.0, 2.0], [3.0, 4.0]])
    values = np.array([[4.0, 5.0]])
    expected, contributions = linear_shap_values(DummyModel(), background, values)
    log_odds = DummyModel.intercept_[0] + values[0] @ DummyModel.coef_[0]
    assert np.isclose(expected + contributions[0].sum(), log_odds)
