"""Training and evaluation primitives used by all public experiments."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from ..evaluation.calibration import fit_temperature, probabilities_from_logits
from ..evaluation.efficiency import synchronize
from ..evaluation.metrics import metrics_from_logits
from ..models import build_model
from ..models.registry import MethodSpec
from ..public_benchmark.cwru_data import CWRUDataset


@dataclass
class TrainState:
    model: nn.Module
    temperature: float
    best_val_f1: float
    best_epoch: int
    epochs_ran: int
    last_train_loss: float
    best_val_loss: float
    history: list[dict[str, float]] = field(default_factory=list)


class WindowSet(Dataset):
    """Index-mapped tensors for one partition."""

    def __init__(
        self,
        data: CWRUDataset,
        indices: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray | None = None,
        augment: bool = False,
    ) -> None:
        self.raw = torch.from_numpy(data.raw)
        self.cwt = torch.from_numpy(data.cwt)
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.weights = torch.as_tensor(
            np.ones(len(indices), dtype=np.float32) if weights is None else weights,
            dtype=torch.float32,
        )
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self,
        local_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        global_index = int(self.indices[local_index])
        raw = self.raw[global_index].float()
        cwt = self.cwt[global_index].float()
        if self.augment:
            shift = int(torch.randint(-32, 33, (1,)).item())
            raw = torch.roll(raw, shifts=shift, dims=-1)
            cwt = torch.roll(cwt, shifts=shift // 16, dims=-1)
            if torch.rand(1).item() < 0.5:
                scale = 0.005 + 0.01 * torch.rand(1).item()
                raw = raw + torch.randn_like(raw) * scale
                cwt = torch.clamp(cwt + torch.randn_like(cwt) * 0.01, min=0.0)
        return raw, cwt, self.labels[local_index], self.weights[local_index], local_index


def forward_model(
    model: nn.Module,
    raw: torch.Tensor,
    cwt: torch.Tensor,
) -> torch.Tensor:
    return model(raw, cwt)


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    local_index_parts: list[np.ndarray] = []
    for raw, cwt, labels, _, local_indices in loader:
        logits = forward_model(model, raw.to(device), cwt.to(device))
        logits_parts.append(logits.float().cpu().numpy())
        label_parts.append(labels.numpy())
        local_index_parts.append(local_indices.numpy())
    return (
        np.concatenate(logits_parts),
        np.concatenate(label_parts),
        np.concatenate(local_index_parts),
    )


def train_model(
    data: CWRUDataset,
    train_indices: np.ndarray,
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    val_indices: np.ndarray,
    method_spec: MethodSpec,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> TrainState:
    model = build_model(
        method_spec.model_key,
        num_classes=int(data.labels.max()) + 1,
        raw_channels=data.raw.shape[1],
        image_channels=data.cwt.shape[1],
    ).to(device)
    train_set = WindowSet(
        data,
        train_indices,
        train_labels,
        train_weights,
        augment=True,
    )
    val_set = WindowSet(data, val_indices, data.labels[val_indices])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    best_epoch = 0
    best_val_loss = float("inf")
    last_train_loss = float("nan")
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        for raw, cwt, labels, weights, _ in train_loader:
            raw = raw.to(device, non_blocking=True)
            cwt = cwt.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = forward_model(model, raw, cwt)
                losses = F.cross_entropy(logits, labels, reduction="none")
                loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_weight = float(weights.sum().detach().cpu())
            total_loss += float(loss.detach().cpu()) * batch_weight
            total_weight += batch_weight
        scheduler.step()
        last_train_loss = total_loss / max(total_weight, 1e-6)

        val_logits, val_labels, _ = predict(model, val_loader, device)
        val_metrics = metrics_from_logits(val_logits, val_labels)
        val_f1 = val_metrics["macro_f1"]
        val_loss = float(
            F.cross_entropy(
                torch.from_numpy(val_logits),
                torch.from_numpy(val_labels),
            )
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(last_train_loss),
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_f1,
            }
        )
        if val_f1 > best_f1 + 1e-5:
            best_f1 = val_f1
            best_epoch = epoch
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    val_logits, val_labels, _ = predict(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    return TrainState(
        model=model,
        temperature=temperature,
        best_val_f1=best_f1,
        best_epoch=best_epoch,
        epochs_ran=len(history),
        last_train_loss=last_train_loss,
        best_val_loss=best_val_loss,
        history=history,
    )


def select_pseudo_labels(
    model: nn.Module,
    data: CWRUDataset,
    candidate_indices: np.ndarray,
    temperature: float,
    max_per_class: int,
    confidence_threshold: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = WindowSet(data, candidate_indices, data.labels[candidate_indices])
    loader = DataLoader(
        dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    logits, _, local_indices = predict(model, loader, device)
    probabilities = probabilities_from_logits(logits, temperature)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    selected_positions: list[int] = []
    num_classes = int(data.labels.max()) + 1
    for class_id in range(num_classes):
        positions = np.flatnonzero(predicted == class_id)
        positions = positions[confidence[positions] >= confidence_threshold]
        if len(positions) == 0:
            continue
        order = positions[np.argsort(confidence[positions])[::-1]]
        selected_positions.extend(order[:max_per_class].tolist())

    selected_positions_array = np.asarray(selected_positions, dtype=np.int64)
    selected_local_indices = local_indices[selected_positions_array]
    return (
        candidate_indices[selected_local_indices].astype(np.int64),
        predicted[selected_positions_array].astype(np.int64),
        confidence[selected_positions_array].astype(np.float32),
        predicted.astype(np.int64),
    )


def run_self_training(
    data: CWRUDataset,
    labeled_indices: np.ndarray,
    candidate_indices: np.ndarray,
    validation_indices: np.ndarray,
    method_spec: MethodSpec,
    *,
    rounds: int,
    confidence_threshold: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> tuple[TrainState, dict[str, float | int]]:
    labels = data.labels[labeled_indices]
    weights = np.ones(len(labeled_indices), dtype=np.float32)
    state = train_model(
        data,
        labeled_indices,
        labels,
        weights,
        validation_indices,
        method_spec,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        num_workers=num_workers,
        device=device,
        seed=seed,
    )
    num_classes = int(data.labels.max()) + 1
    max_per_class = max(1, len(labeled_indices) // num_classes * 2)
    pseudo_count = 0
    pseudo_accuracy = float("nan")
    pseudo_macro_f1 = float("nan")
    pseudo_coverage = 0.0
    rounds_completed = 0
    for round_id in range(1, rounds + 1):
        pseudo_idx, pseudo_labels, pseudo_weights, _ = select_pseudo_labels(
            state.model,
            data,
            candidate_indices,
            state.temperature,
            max_per_class=max_per_class,
            confidence_threshold=confidence_threshold,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        pseudo_count = len(pseudo_idx)
        pseudo_coverage = pseudo_count / max(len(candidate_indices), 1)
        if pseudo_count == 0:
            break
        pseudo_accuracy = float(accuracy_score(data.labels[pseudo_idx], pseudo_labels))
        pseudo_macro_f1 = float(
            f1_score(
                data.labels[pseudo_idx],
                pseudo_labels,
                average="macro",
                zero_division=0,
            )
        )
        combined_idx = np.concatenate([labeled_indices, pseudo_idx])
        combined_labels = np.concatenate([labels, pseudo_labels])
        combined_weights = np.concatenate([weights, pseudo_weights])
        state = train_model(
            data,
            combined_idx,
            combined_labels,
            combined_weights,
            validation_indices,
            method_spec,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            num_workers=num_workers,
            device=device,
            seed=seed + round_id,
        )
        rounds_completed = round_id
        max_per_class *= 2
    return state, {
        "pseudo_count": pseudo_count,
        "pseudo_coverage": pseudo_coverage,
        "pseudo_accuracy": pseudo_accuracy,
        "pseudo_macro_f1": pseudo_macro_f1,
        "self_training_rounds_completed": rounds_completed,
        "self_training_round_limit": rounds,
    }


def evaluate_model(
    state: TrainState,
    data: CWRUDataset,
    test_indices: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[
    dict[str, float],
    dict[str, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float],
]:
    test_set = WindowSet(data, test_indices, data.labels[test_indices])
    loader = DataLoader(
        test_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    state.model.eval()
    synchronize(device)
    start = time.perf_counter()
    logits, labels, _ = predict(state.model, loader, device)
    synchronize(device)
    inference_time_ms = float((time.perf_counter() - start) * 1000)
    raw_metrics = metrics_from_logits(logits, labels, temperature=1.0)
    calibrated_metrics = metrics_from_logits(
        logits,
        labels,
        temperature=state.temperature,
    )
    predictions = probabilities_from_logits(
        logits,
        state.temperature,
    ).argmax(axis=1)
    timing = {
        "inference_time_ms": inference_time_ms,
        "inference_batches": float(len(loader)),
        "inference_ms_per_batch": inference_time_ms / max(len(loader), 1),
        "inference_samples_per_s": (
            len(test_indices) / max(inference_time_ms / 1000, 1e-12)
        ),
    }
    return (
        raw_metrics,
        calibrated_metrics,
        logits,
        labels,
        predictions,
        timing,
    )
