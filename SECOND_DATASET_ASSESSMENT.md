# Second Dataset Assessment

## Decision

No second public dataset in the reviewed material was added to the primary
benchmark. None simultaneously provides:

1. enough repeated physical units within each target fault class;
2. at least two explicitly identifiable operating conditions;
3. a unit-and-condition-disjoint evaluation that leaves the target classes
   represented; and
4. a documented mapping from recordings to physical bearings.

Adding a dataset that fails these conditions would create a second result with a
different estimand and would not provide an external-validity test of the
current protocol.

## Candidate assessment

| Candidate | Physical-unit support | Operating-condition support | Unit-disjoint feasibility | Decision |
| --- | --- | --- | --- | --- |
| Paderborn University Bearing Data Center | 16 reviewed codes; the primary subset has 4 held-out codes | Four explicit speed, torque, and radial-force settings | Yes, with simultaneous condition holdout | Primary dataset |
| CWRU Bearing Data Center | Fault-bearing identity is inferred from label/config; public documentation does not establish a distinct physical unit for every load and fault configuration | Four motor loads | Unit identity is not sufficiently documented for a strict replacement benchmark | Exclude |
| HUST Bearing | One bearing per fault-state and bearing-type combination is assumed; the same bearing is reused across loads | Three motor loads | Classes could be held out, but repeated physical units within a fault class are not available | Exclude from primary external-validity claim |
| HUST Transmission | One left and one right bearing on a shared transmission rig | Seven constant speeds plus run-up/run-down | Too few physical bearings per fault location; compound shifts change several components at once | Exclude |
| University of Ottawa Rolling-Element Dataset | Five bearings per inner-, outer-, ball-, and cage-fault family, recorded at healthy, developing, and faulty stages | Constant load and speed in the reviewed version | Physical-unit holdout is possible, but an operating-condition holdout is not | Exclude as a condition-shift check |
| NASA IMS | Four bearings per run with documented failed elements | Fixed speed and load | Natural run-to-failure labels and timing differ from the four-class task; no explicit condition holdout | Exclude |

## Minimum conditions for a future second benchmark

A suitable extension should name the physical unit for every recording,
duplicate units within every target class, report all operating conditions, and
predeclare folds that remove both unit and condition from fitting. If those
conditions are met, the efficient extension is to run Raw 1D-CNN, CWT 2D-CNN,
and CWT-GRU only, with the same metrics and fold-first inference. The full DRSN,
TFN, calibration, and self-training matrix should not be repeated unless the
new dataset supports the same estimand.

## Sources reviewed

- Paderborn Bearing Data Center:
  <https://groups.uni-paderborn.de/kat/BearingDataCenter/>
- CWRU:
  <https://engineering.case.edu/bearingdatacenter>
- HUST Bearing:
  <https://data.mendeley.com/datasets/cbv7jyx4p9/3>
- HUST Transmission:
  <https://github.com/CHAOZHAO-1/HUSTTransmissionsystem-dataset>
- University of Ottawa rolling-element data:
  <https://data.mendeley.com/datasets/y2px5tg92h/5>
- NASA IMS:
  <https://data.nasa.gov/dataset/ims-bearings>
- Dataset metadata cross-check:
  `VictorBauler/BearingDatasets` repository,
  `src/bearing_datasets/datasets/*/dataset.yaml`.

This assessment concerns availability and protocol compatibility. It does not
compare the scientific quality or usefulness of the excluded datasets.
