# Dataset

## Source and redistribution

The benchmark uses the Paderborn University Bearing Data Center. Original
signals and generated caches are not redistributed.

- Official source:
  <https://groups.uni-paderborn.de/kat/BearingDataCenter/>
- Archive URL pattern:
  `https://groups.uni-paderborn.de/kat/BearingDataCenter/<bearing_code>.rar`
- Archive byte sizes and SHA-256 values:
  `data/manifests/paderborn_archives_sha256.csv`
- Selected bearing manifest:
  `data/manifests/paderborn_selected_bearings.csv`

Users must obtain the data from the official source and follow its terms.

## Inputs

- Signals: vibration channel 1 and phase currents 1 and 2.
- Sampling rate: 64 kHz.
- Measurements: files 1 and 2 for each selected bearing and condition.
- Window length: 8192 samples.
- Stride: 8192 samples, so adjacent windows do not overlap.
- Canonical tensor: 3,939 windows with raw shape `3939 x 3 x 8192` and CWT
  shape `3939 x 3 x 96 x 96`.

The CWT representation uses the Morlet wavelet, 48 geometric scales, magnitude
only, a 96 x 96 image, and per-channel maximum normalization.

## Physical bearings

The 12 development bearings are:

```text
K001 K002 K003 K004 KA01 KA04 KA08 KI01 KI04 KI08 KB23 KB24
```

The four held-out test bearings are:

```text
K005 KA15 KI14 KB27
```

No held-out bearing enters training, validation, calibration fitting,
pseudo-label selection, or model selection.

## Operating conditions

```text
C0 = N09_M07_F10
C1 = N15_M01_F10
C2 = N15_M07_F04
C3 = N15_M07_F10
```

Each fold holds out one condition for test and a different condition for
validation. The condition mapping is frozen in
`results/splits/fold0.json` through `fold3.json` and indexed in
`splits/manifest.json`.

## Labels and damage origin

The four classes are:

```text
Healthy
Outer-ring damage
Inner-ring damage
Multiple damage
```

Some development bearings were artificially damaged, while held-out test
bearings include lifetime damage. Bearing identity, operating condition, and
damage origin are therefore confounded in the primary holdout.

The complete bearing/class/origin mapping is in
`results/damage_origin_manifest.csv`.

## Canonical cache

```powershell
python scripts/prepare_data.py
```

The command downloads and verifies the selected official archives, extracts
measurement files 1 and 2, and writes:

```text
data/public/paderborn/pu_4class_w8192.npz
```

The recorded cache SHA-256 is:

```text
a898abcb63e17ee68f2014d19b4d0f675b6c4d19a9abc89376031a78c3a23df4
```

If archives are already present, use:

```powershell
python scripts/prepare_data.py --skip-download
```

## Original-amplitude audit

The raw-amplitude audit cache preserves original amplitudes, source filenames,
measurements, and window starts:

```powershell
python -m spindle_cnn.public_benchmark.paderborn_raw_windows `
  --data-root data/public/paderborn/extracted `
  --output data/public/paderborn/pu_4class_w8192_raw.npz `
  --validate
```

Re-standardization reproduces the canonical cache elementwise. One official
file, `N15_M01_F10_KA08_2.mat`, is truncated; the canonical and raw-audit
preprocessing paths skip the same unreadable measurement.

## CWRU

CWRU is retained only as a pilot/saturation check because it does not provide
the same physical-bearing grouping needed for the primary protocol. CWRU
leave-one-load-out results must not be described as bearing-code-disjoint.
