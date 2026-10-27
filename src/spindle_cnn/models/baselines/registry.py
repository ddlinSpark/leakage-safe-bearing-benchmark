"""Explicit placeholders for baselines that are not yet implemented.

No placeholder is available to the model factory. A baseline must implement the
same ``BaseModel`` contract and pass a same-split reproduction check before it
can be added to a paper table.
"""

from __future__ import annotations


PLACEHOLDER_BASELINES: dict[str, str] = {
    "cnn_lstm": "Architecture, preprocessing, and optimization not yet reproduced.",
    "drsn_transformer": "Original architecture and training recipe not yet reproduced.",
    "mtfc_t": "MTFC-T definition and data pipeline not yet reproduced.",
}


def available_placeholders() -> dict[str, str]:
    """Return unavailable methods and the reason each remains excluded."""

    return dict(PLACEHOLDER_BASELINES)
