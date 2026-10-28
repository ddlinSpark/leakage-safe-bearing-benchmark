# Paderborn Metadata Manifests

`paderborn_selected_bearings.csv` records the physical-bearing, fault-location,
damage-origin, damage-method, severity, manufacturer, and split-role metadata
used by the damage-origin audit. The values come from the Paderborn bearing fact
sheets summarized in `VictorBauler/BearingDatasets` at commit
`1ca6bf36937555e4b4e91619a21a4eefd5d0ad0c` (the checkout used during this
revision). The source dataset remains subject to its own CC BY-NC 4.0 terms.

`paderborn_archives_sha256.csv` records the exact archive sizes and SHA-256
digests used to reconstruct the raw-amplitude audit cache. Raw archives and
large caches are not redistributed.

The executable audit reads only the 16 selected bearing codes. Facts not needed
to establish the split, label, or damage-origin audit are intentionally omitted.
