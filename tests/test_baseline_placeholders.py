from __future__ import annotations

from spindle_cnn.models.baselines import available_placeholders
from spindle_cnn.models.registry import METHODS


def test_unreproduced_sota_names_are_not_available_as_models() -> None:
    placeholders = available_placeholders()
    assert set(placeholders) == {"cnn_lstm", "drsn_transformer", "mtfc_t"}
    assert not set(placeholders) & set(METHODS)
