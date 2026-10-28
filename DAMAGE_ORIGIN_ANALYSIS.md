# Damage-Origin Analysis

## Finding

The primary validation-to-test gap coincides with joint changes in physical bearing identity, operating condition, and damage origin. The current four-class protocol cannot attribute the gap to one of these factors.

All held-out fault bearings have real (accelerated-lifetime) damage. Artificial damage appears only in the fitting set for outer- and inner-ring bearings. Both fitting multiple-damage bearings are real-damage bearings. The held-out healthy bearing is not subject to a damage-origin change.

## Held-out origin

| Bearing | Class | Origin |
| --- | --- | --- |
| K005 | Healthy | none |
| KA15 | OuterRing | real |
| KB27 | MultipleDamage | real |
| KI14 | InnerRing | real |

## Bearing codes by class and origin

| Class | Origin | Bearing codes |
| --- | --- | ---: |
| Healthy | none | 5 |
| InnerRing | artificial | 2 |
| InnerRing | real | 2 |
| MultipleDamage | real | 3 |
| OuterRing | artificial | 2 |
| OuterRing | real | 2 |

## Window distribution

| Split | Origin | Windows | Bearings represented |
| --- | --- | ---: | ---: |
| test | none | 248 | 1 |
| test | real | 744 | 3 |
| train | artificial | 1922 | 4 |
| train | none | 1986 | 4 |
| train | real | 1986 | 4 |
| validation | artificial | 961 | 4 |
| validation | none | 993 | 4 |
| validation | real | 993 | 4 |

## Feasibility of damage-origin-only controls

| Proposed control | Feasible with selected 16 codes? | Reason |
| --- | --- | --- |
| Artificial to artificial, four classes | No | No artificial multiple-damage bearing is available. |
| Lifetime to lifetime, four classes | Partial only | Four classes exist, but single-location lifetime groups contain only one training bearing each. |
| Artificial to lifetime, four classes | No | All held-out fault bearings are lifetime-damaged and no artificial multiple-damage source exists. |
| Artificial to lifetime, three classes | Possible but strongly reduced | It can compare healthy, outer, and inner classes after removing multiple damage, but source diversity is low. |

A one-origin holdout would also change bearing identity and condition. It would not create a clean causal estimate. The study therefore retains the joint-shift interpretation and reports the origin distribution as a confounding audit rather than converting it into a single-factor claim.

## Result provenance

- `results/damage_origin_manifest.csv`
- `results/damage_origin_distribution.csv`
- `reports/label_audit.csv`
- `data/manifests/paderborn_selected_bearings.csv`
