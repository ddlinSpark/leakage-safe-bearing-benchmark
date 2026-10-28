"""Verify centralized palette adoption and publication figure integrity."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image

from spindle_cnn.visualization.style import MODEL_COLORS, PAPER_COLORS


ROOT = Path(__file__).resolve().parents[1]
PLOT_MODULES = [
    "src/spindle_cnn/analysis/baseline_tuning.py",
    "src/spindle_cnn/analysis/calibration_audit.py",
    "src/spindle_cnn/analysis/cwt_sensitivity.py",
    "src/spindle_cnn/analysis/domain_shift.py",
    "src/spindle_cnn/analysis/feature_space.py",
    "src/spindle_cnn/analysis/foldwise.py",
    "src/spindle_cnn/analysis/label_granularity.py",
    "src/spindle_cnn/analysis/normalization_sensitivity.py",
    "src/spindle_cnn/analysis/self_training.py",
    "src/spindle_cnn/analysis/training_budget.py",
    "src/spindle_cnn/experiments/make_figures.py",
    "src/spindle_cnn/experiments/make_method_figures.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    publication = ROOT / "figures" / "publication"
    palette = {value.lower() for value in PAPER_COLORS}
    source_issues: list[str] = []
    for relative in PLOT_MODULES:
        text = (ROOT / relative).read_text(encoding="utf-8")
        if "apply_paper_style" not in text:
            source_issues.append(f"{relative}: does not call apply_paper_style")
        local_hex = {
            value.lower()
            for value in re.findall(r"#[0-9A-Fa-f]{6}", text)
        }
        extra = sorted(local_hex - palette)
        if extra:
            source_issues.append(f"{relative}: local palette colors {extra}")

    pngs = sorted(publication.glob("*.png"))
    pdfs = sorted(publication.glob("*.pdf"))
    stems_png = {path.stem for path in pngs}
    stems_pdf = {path.stem for path in pdfs}
    pair_issues = []
    if stems_png != stems_pdf:
        pair_issues.append(
            f"unpaired PNG/PDF stems: {sorted(stems_png ^ stems_pdf)}"
        )
    if len(pngs) != 24 or len(pdfs) != 24:
        pair_issues.append(f"expected 24 PNG/PDF pairs; found {len(pngs)}/{len(pdfs)}")
    raster_issues = []
    for path in pngs:
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))
            if min(dpi) < 299:
                raster_issues.append(f"{path.name}: DPI metadata {dpi}")

    manifest_path = publication / "figure_hashes.json"
    manifest_issues = []
    if not manifest_path.is_file():
        manifest_issues.append("figure_hashes.json is missing")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
        expected_names = {path.name for path in [*pngs, *pdfs]}
        if set(manifest) != expected_names:
            manifest_issues.append("manifest file set differs from publication files")
        for path in [*pngs, *pdfs]:
            record = manifest.get(path.name, {})
            if record.get("bytes") != path.stat().st_size:
                manifest_issues.append(f"{path.name}: byte count mismatch")
            elif record.get("sha256") != sha256(path):
                manifest_issues.append(f"{path.name}: SHA-256 mismatch")

    issues = source_issues + pair_issues + raster_issues + manifest_issues
    payload = {
        "status": "PASS" if not issues else "FAIL",
        "palette": list(PAPER_COLORS),
        "model_colors": MODEL_COLORS,
        "plot_modules_checked": len(PLOT_MODULES),
        "png_count": len(pngs),
        "pdf_count": len(pdfs),
        "source_issues": source_issues,
        "pair_issues": pair_issues,
        "raster_issues": raster_issues,
        "manifest_issues": manifest_issues,
    }
    output = ROOT / "artifacts" / "local" / "figure_palette_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if not issues else 1)


if __name__ == "__main__":
    main()
