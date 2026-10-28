"""Architecture registry and method-to-model mapping."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from .cwt_cnn import CWT2DCNN
from .cwt_gru import CWTGRU
from .drsn import DRSN1D
from .gru import RawGRU
from .raw_cnn import RawCNN
from .tfnet import TFN


@dataclass(frozen=True)
class MethodSpec:
    method: str
    model_key: str
    uses_temperature: bool = False
    uses_self_training: bool = False
    pseudo_strategy: str = "threshold"


METHODS: dict[str, MethodSpec] = {
    "raw_cnn": MethodSpec("raw_cnn", "raw_cnn"),
    "gru_only": MethodSpec("gru_only", "gru_only"),
    "cwt_cnn": MethodSpec("cwt_cnn", "cwt_cnn"),
    "cwt_gru_no_se": MethodSpec("cwt_gru_no_se", "cwt_gru_no_se"),
    "cwt_gru": MethodSpec("cwt_gru", "cwt_gru"),
    "cwt_gru_temp": MethodSpec("cwt_gru_temp", "cwt_gru", uses_temperature=True),
    "cwt_gru_selftrain": MethodSpec(
        "cwt_gru_selftrain",
        "cwt_gru",
        uses_temperature=True,
        uses_self_training=True,
    ),
    "cwt_gru_selftrain_coverage": MethodSpec(
        "cwt_gru_selftrain_coverage",
        "cwt_gru",
        uses_temperature=True,
        uses_self_training=True,
        pseudo_strategy="coverage",
    ),
    "drsn": MethodSpec("drsn", "drsn"),
    "tfnet": MethodSpec("tfnet", "tfnet"),
}


def validate_methods(methods: list[str]) -> None:
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid methods: {sorted(METHODS)}")


def build_model(
    name: str,
    num_classes: int = 10,
    raw_channels: int = 2,
    image_channels: int = 2,
) -> nn.Module:
    if name == "raw_cnn":
        return RawCNN(num_classes=num_classes, in_channels=raw_channels)
    if name == "cwt_cnn":
        return CWT2DCNN(num_classes=num_classes, in_channels=image_channels)
    if name == "cwt_gru":
        return CWTGRU(
            num_classes=num_classes,
            raw_channels=raw_channels,
            image_channels=image_channels,
        )
    if name == "gru_only":
        return RawGRU(num_classes=num_classes, raw_channels=raw_channels)
    if name == "cwt_gru_no_se":
        return CWTGRU(
            num_classes=num_classes,
            raw_channels=raw_channels,
            image_channels=image_channels,
            use_se=False,
        )
    if name == "drsn":
        return DRSN1D(num_classes=num_classes, raw_channels=raw_channels)
    if name == "tfnet":
        return TFN(num_classes=num_classes, in_channels=raw_channels)
    raise ValueError(f"Unknown model: {name}")
