from __future__ import annotations

import pytest
import torch

from spindle_cnn.models import build_model


@pytest.mark.parametrize(
    "model_name",
    [
        "raw_cnn",
        "gru_only",
        "cwt_cnn",
        "cwt_gru_no_se",
        "cwt_gru",
        "drsn",
        "tfnet",
    ],
)
def test_every_registered_model_returns_one_logit_per_class(model_name: str) -> None:
    model = build_model(
        model_name,
        num_classes=4,
        raw_channels=2,
        image_channels=2,
    )
    raw = torch.randn(2, 2, 256)
    cwt = torch.randn(2, 2, 32, 32)
    logits = model(raw, cwt)
    assert logits.shape == (2, 4)


def test_unknown_baseline_is_not_silently_registered() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("unknown_external_model")
