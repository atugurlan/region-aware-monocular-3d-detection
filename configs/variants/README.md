# Variant configs

This folder contains runnable configs for the region-aware ablation stage.

The initial plan was to run V1-V5 in order. After the first experiments, the plan changed slightly. V2 became the main branch because it is the simplest meaningful ROI-grid variant: a 3x3 grid pooled from inside the predicted object box.

## Current recommended order

1. Run the baseline: `configs/monodgp_baseline_20.yaml`.
2. Use V2 as the main region-aware branch.
3. Tune the regional geometry loss on V2:
   - `v2_grid3x3.yaml`: coefficient `0.2`.
   - `v2_grid3x3_loss005.yaml`: coefficient `0.05`.
   - `v2_grid3x3_loss01.yaml`: coefficient `0.1`.
4. Keep the best V2 tuning as the reference region-aware model.
5. Only after that, test uncertainty and mask variants.

## Config list

| Config | Meaning | Notes |
| --- | --- | --- |
| `v1_grid2x2.yaml` | ROI grid 2x2 regional depth correction | optional grid-size ablation |
| `v2_grid3x3.yaml` | ROI grid 3x3 regional depth correction, loss coefficient 0.2 | stable with gate, but below baseline |
| `v2_grid3x3_loss005.yaml` | V2 with loss coefficient 0.05 | current best result |
| `v2_grid3x3_loss01.yaml` | V2 with loss coefficient 0.1 | intermediate tuning run |
| `v3_grid3x3_uncertainty.yaml` | V2 + depth uncertainty weighting | next extension candidate |
| `v4_grid3x3_mask.yaml` | V2 + region-mask guidance | next extension candidate |
| `v5_grid3x3_uncertainty_mask.yaml` | V2 + uncertainty + mask | only useful after V3/V4 are checked |

All current variants use ROIAlign on the predicted 2D object box. The grid is pooled from inside the object ROI, not globally from the full feature map.

The regional depth correction is multiplied by a learnable gate initialized at zero. This keeps the starting behavior close to the original MonoDGP baseline and lets the model learn whether the regional correction is useful.

For fair comparison, use the same dataset split, the same epoch count, and the same visualization image IDs.
