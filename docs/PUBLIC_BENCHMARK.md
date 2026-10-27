# Public Benchmark Protocol

The revised study separates model evidence from rule-generated industrial labels.
Only datasets with verifiable labels are used for accuracy claims.

## CWRU protocol

- Source: official CWRU Bearing Data Center, 12 kHz drive-end data.
- Sensors: drive-end and fan-end accelerometers.
- Classes: healthy and ball, inner-race, and outer-race faults at three
  diameters, giving 10 classes.
- Windowing: 2048 samples with a stride of 2048; adjacent windows do not
  overlap.
- Evaluation: leave-one-load-out. Training and validation use different loads
  from the test fold, and test labels are never used for training or model
  selection.
- Repeated runs: multiple random seeds; mean, standard deviation, 95%
  confidence interval, and paired comparisons are reported.

CWRU is retained as a saturation check rather than the sole evidence for the
proposed architecture. Public pilot results show that a raw dual-channel CNN can
already approach perfect accuracy under this split, so a more complex model is
not claimed to be superior on CWRU alone.

## Paderborn protocol

- Source: Paderborn University Bearing Data Center.
- Sensors: vibration and two phase-current channels at 64 kHz.
- Classes: healthy, outer-ring damage, inner-ring damage, and multiple damage.
- Training bearings are K001-K004, KA01/KA04/KA08, KI01/KI04/KI08, and
  KB23/KB24. K005, KA15, KI14, and KB27 are reserved for testing.
- Operating conditions are grouped; each fold leaves one condition out.
- Windows are 8192 samples with no overlap.
- The strict test set combines unseen bearing codes with an unseen operating
  condition. Validation uses a different seen-bearing condition.

## Reproducing the experiments

```bash
spindle-cwru-download
spindle-cwru-cache
python -m spindle_cnn.public_benchmark.train_cwru \
  --cache data/public/cwru/cwru_10class_w2048_morl.npz \
  --output-dir outputs/public_cwru \
  --methods raw_cnn,cwt_cnn,cwt_gru,cwt_gru_selftrain \
  --fractions 0.1,0.25,0.5,1.0 \
  --folds 0,1,2,3 --seeds 42,43,44
spindle-public-summary --results outputs/public_cwru/results.csv \
  --output-dir outputs/public_cwru/analysis
```

Paderborn requires downloading the relevant archives from the official data
center and extracting the first five measurement files under each selected
bearing and operating condition. The local archive is not redistributed.

```bash
spindle-pu-cache
python -m spindle_cnn.public_benchmark.train_cwru \
  --dataset-name Paderborn-4class \
  --cache data/public/paderborn/pu_4class_w8192.npz \
  --output-dir outputs/public_paderborn \
  --heldout-record-ids 4,8,12,15 \
  --methods raw_cnn,cwt_cnn,cwt_gru,cwt_gru_selftrain \
  --fractions 0.1,0.25,0.5,1.0 \
  --folds 0,1,2,3 --seeds 42,43,44
spindle-public-summary --results outputs/public_paderborn/results.csv \
  --output-dir outputs/public_paderborn/analysis
```
