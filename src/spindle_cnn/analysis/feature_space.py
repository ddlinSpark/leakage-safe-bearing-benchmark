"""Post-hoc penultimate-feature visualization and clustering diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ..datasets.splits import balanced_indices, load_fold_indices, set_global_seed
from ..models.registry import METHODS
from ..paths import PROJECT_ROOT
from ..public_benchmark.cwru_data import CWRUDataset, load_cwru_cache
from ..public_benchmark.pu_data import CLASS_NAMES, RECORD_ID_TO_BEARING
from ..training.runner import WindowSet, train_model


def extract_embeddings(
    model: torch.nn.Module,
    data: CWRUDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = WindowSet(data, indices, data.labels[indices])
    loader = DataLoader(dataset, batch_size=batch_size * 2, shuffle=False)
    model.eval()
    parts = []
    with torch.inference_mode():
        for raw, cwt, _, _, _ in loader:
            parts.append(
                model.embedding(raw.to(device), cwt.to(device))
                .float()
                .cpu()
                .numpy()
            )
    return np.concatenate(parts)


def safe_silhouette(values: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2 or len(labels) <= len(np.unique(labels)):
        return float("nan")
    return float(silhouette_score(values, labels))


def neighbour_agreement(
    values: np.ndarray,
    classes: np.ndarray,
    bearings: np.ndarray,
    conditions: np.ndarray,
    *,
    k: int = 5,
) -> dict[str, float]:
    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(values)
    _, indices = neighbours.kneighbors(values)
    neighbours = indices[:, 1:]
    return {
        "nearest_same_class_fraction": float(
            (classes[neighbours] == classes[:, None]).mean()
        ),
        "nearest_same_bearing_fraction": float(
            (bearings[neighbours] == bearings[:, None]).mean()
        ),
        "nearest_same_condition_fraction": float(
            (conditions[neighbours] == conditions[:, None]).mean()
        ),
    }


def analyze_fold(
    data: CWRUDataset,
    *,
    fold: int,
    seed: int,
    device: torch.device,
    epochs: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    split = load_fold_indices(data, fold, np.asarray([4, 8, 12, 15]))
    set_global_seed(seed)
    rng = np.random.default_rng(seed + fold * 1000)
    labeled = balanced_indices(split.train, data.labels, 1.0, rng)
    state = train_model(
        data,
        labeled,
        data.labels[labeled],
        np.ones(len(labeled), dtype=np.float32),
        split.validation,
        METHODS["cwt_gru"],
        epochs=epochs,
        batch_size=64,
        learning_rate=0.002,
        weight_decay=0.0001,
        patience=4,
        num_workers=0,
        device=device,
        seed=seed,
    )
    arrays: dict[str, np.ndarray] = {}
    all_indices = []
    all_embeddings = []
    split_names = []
    for split_name, indices in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        embeddings = extract_embeddings(
            state.model,
            data,
            indices,
            batch_size=64,
            device=device,
        )
        arrays[f"{split_name}_indices"] = indices.astype(np.int64)
        arrays[f"{split_name}_embeddings"] = embeddings.astype(np.float32)
        all_indices.append(indices)
        all_embeddings.append(embeddings)
        split_names.extend([split_name] * len(indices))
    all_indices_array = np.concatenate(all_indices)
    embeddings = np.concatenate(all_embeddings)
    scaled = StandardScaler().fit_transform(embeddings)
    pca = PCA(n_components=min(20, scaled.shape[0] - 1, scaled.shape[1]))
    pca_values = pca.fit_transform(scaled)
    domains = np.asarray(split_names)
    classes = data.labels[all_indices_array]
    bearings = np.asarray(
        [RECORD_ID_TO_BEARING[int(value)] for value in data.record_ids[all_indices_array]]
    )
    conditions = np.asarray(
        [f"C{int(value)}" for value in data.loads[all_indices_array]]
    )
    test_mask = domains == "test"
    agreement = neighbour_agreement(
        pca_values,
        classes,
        bearings,
        conditions,
    )
    metrics = {
        "fold": fold,
        "seed": seed,
        "train_windows": len(split.train),
        "validation_windows": len(split.validation),
        "test_windows": len(split.test),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "domain_silhouette": safe_silhouette(pca_values, domains),
        "test_class_silhouette": safe_silhouette(
            pca_values[test_mask],
            classes[test_mask],
        ),
        "all_bearing_silhouette": safe_silhouette(pca_values, bearings),
        "all_condition_silhouette": safe_silhouette(pca_values, conditions),
        **agreement,
    }
    arrays["pca_2d"] = pca_values[:, :2].astype(np.float32)
    arrays["indices"] = all_indices_array
    arrays["split"] = domains
    arrays["classes"] = classes
    arrays["bearings"] = bearings
    arrays["conditions"] = conditions
    return metrics, arrays


def save_figures(
    fold_arrays: dict[int, dict[str, np.ndarray]],
    metrics: pd.DataFrame,
    output_dir: Path,
    *,
    tsne_fold: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    markers = {"train": "o", "validation": "s", "test": "^"}
    colors = {"train": "#D0D0D0", "validation": "#808080", "test": "#111111"}
    for fold, axis in zip(sorted(fold_arrays), axes.ravel(), strict=True):
        arrays = fold_arrays[fold]
        for split_name in ("train", "validation", "test"):
            mask = arrays["split"] == split_name
            axis.scatter(
                arrays["pca_2d"][mask, 0],
                arrays["pca_2d"][mask, 1],
                s=9 if split_name != "test" else 15,
                marker=markers[split_name],
                facecolors=colors[split_name],
                edgecolors="none",
                alpha=0.70,
                label=split_name.capitalize(),
            )
        axis.set_title(f"Fold {fold}", loc="left")
        axis.set_xlabel("PC 1")
        axis.set_ylabel("PC 2")
        axis.grid(color="0.92", linewidth=0.6)
        axis.legend(frameon=False, fontsize=6.5, markerscale=0.8)
    png = output_dir / "figure_22_feature_space_pca.png"
    pdf = output_dir / "figure_22_feature_space_pca.pdf"
    figure.savefig(png, dpi=320, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    arrays = fold_arrays[tsne_fold]
    sample_limit = 1200
    rng = np.random.default_rng(20261028)
    selected = np.arange(len(arrays["pca_2d"]))
    if len(selected) > sample_limit:
        selected = rng.choice(selected, sample_limit, replace=False)
    manifold = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=20261028,
    ).fit_transform(arrays["pca_2d"][selected])
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
    class_colors = ["#202020", "#6F6F6F", "#A8A8A8", "#000000"]
    class_markers = ["o", "s", "^", "D"]
    for class_id, name in enumerate(CLASS_NAMES):
        mask = arrays["classes"][selected] == class_id
        axes[0].scatter(
            manifold[mask, 0],
            manifold[mask, 1],
            s=10,
            color=class_colors[class_id],
            marker=class_markers[class_id],
            edgecolors="none",
            label=name,
            alpha=0.8,
        )
    axes[0].set_title("(a) Fault-class colouring", loc="left")
    test_bearings = sorted(
        set(arrays["bearings"][selected][arrays["split"][selected] == "test"])
    )
    for index, bearing in enumerate(test_bearings):
        mask = (arrays["bearings"][selected] == bearing) & (
            arrays["split"][selected] == "test"
        )
        axes[1].scatter(
            manifold[mask, 0],
            manifold[mask, 1],
            s=14,
            color=class_colors[index],
            edgecolors="black",
            linewidths=0.2,
            label=bearing,
        )
    axes[1].set_title("(b) Held-out bearing colouring", loc="left")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.legend(frameon=False, fontsize=6.5)
    axes[0].set_ylabel(f"t-SNE dimension 2 (fold {tsne_fold})")
    png = output_dir / "figure_23_feature_space_tsne.png"
    pdf = output_dir / "figure_23_feature_space_tsne.pdf"
    figure.savefig(png, dpi=320, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "paderborn_main.json",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "publication",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--tsne-fold", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import json

    config = json.loads(args.config.read_text(encoding="utf-8"))
    cache = (PROJECT_ROOT / config["dataset"]["cache"]).resolve()
    data = load_cwru_cache(cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dir = args.results_dir / "feature_embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    fold_arrays = {}
    for fold in range(4):
        fold_metrics, arrays = analyze_fold(
            data,
            fold=fold,
            seed=args.seed,
            device=device,
            epochs=args.epochs,
        )
        metrics.append(fold_metrics)
        fold_arrays[fold] = arrays
        np.savez_compressed(embedding_dir / f"fold{fold}.npz", **arrays)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(
        args.results_dir / "feature_space_clustering.csv",
        index=False,
        lineterminator="\n",
    )
    save_figures(
        fold_arrays,
        metric_frame,
        args.figures_dir,
        tsne_fold=args.tsne_fold,
    )
    print(metric_frame.to_string(index=False))


if __name__ == "__main__":
    main()
