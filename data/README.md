# Public Data

The repository does not redistribute raw or preprocessed Paderborn/CWRU data.
The local data and cache directories are excluded from Git.

**Raw data are not redistributed.**

## Paderborn University Bearing Data Center

- Official data center:
  <https://groups.uni-paderborn.de/kat/BearingDataCenter/>
- Archive URL pattern:
  `https://groups.uni-paderborn.de/kat/BearingDataCenter/<bearing_code>.rar`
- Main reference: C. Lessmeier, J. K. Kimotho, D. Zimmer, and W. Sextro,
  "Condition monitoring of bearing damage in electromechanical drive systems by
  using motor current signals of electric motors: A benchmark data set for
  data-driven classification," 2016,
  doi: 10.36001/phme.2016.v3i1.1577.
- Signals used: `vibration_1`, `phase_current_1`, and `phase_current_2`.
- Sampling rate: 64 kHz.
- Window size: 8192 samples.
- Stride: 8192 samples; windows do not overlap.
- Classes: healthy, outer-ring damage, inner-ring damage, and multiple damage.
- Training bearing codes: `K001`, `K002`, `K003`, `K004`, `KA01`, `KA04`,
  `KA08`, `KI01`, `KI04`, `KI08`, `KB23`, and `KB24`.
- Validation bearing codes: the same seen training codes; the four held-out
  test codes are excluded.
- Unseen test bearing codes: `K005`, `KA15`, `KI14`, and `KB27`.
- Operating conditions: `N09_M07_F10`, `N15_M01_F10`, `N15_M07_F04`, and
  `N15_M07_F10`.
- Validation/test split: one operating condition is held out as test; a second
  is used for validation; the other two are used for training.
- Measurements: files 1 and 2 for every selected bearing and condition.

Download and preprocess:

```bash
PYTHONPATH=src python -m spindle_cnn.public_benchmark.prepare_paderborn
PYTHONPATH=src python -m spindle_cnn.public_benchmark.pu_data \
  --data-root data/public/paderborn/extracted \
  --output data/public/paderborn/pu_4class_w8192.npz
```

The extraction command requires a local archive extractor and the configured
Paderborn archives. Pass `--seven-zip` with the local 7-Zip executable or set
the `SEVEN_ZIP` environment variable. See `prepare_paderborn.py` for its CLI
options.

## CWRU Bearing Data Center

- Official data center:
  <https://engineering.case.edu/bearingdatacenter>
- Sampling rate: 12 kHz drive-end data.
- Window size: 2048 samples.
- Stride: 2048 samples.
- Evaluation: leave-one-load-out.
- Role: pilot/saturation check only. CWRU is not the paper's primary
  bearing-code-disjoint evidence.

```bash
PYTHONPATH=src python -m spindle_cnn.public_benchmark.download_cwru
PYTHONPATH=src python -m spindle_cnn.public_benchmark.cwru_data
```

Consult each data provider's terms before redistributing any signal or derived
tensor.
