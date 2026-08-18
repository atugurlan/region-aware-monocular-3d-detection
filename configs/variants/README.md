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
6. Treat V6 as an adaptive-fusion ablation.
7. Use V7.1 as the first reliability experiment, starting from the stable V2.1 path.
8. Use V7.2 only after V7.1, because V7.2 lets reliability change the region weights directly.

## Config list

| Config | Meaning | Notes |
| --- | --- | --- |
| `v1_grid2x2.yaml` | ROI grid 2x2 regional depth correction | optional grid-size ablation |
| `v2_grid3x3.yaml` | ROI grid 3x3 regional depth correction, loss coefficient 0.2 | stable with gate, but below baseline |
| `v2_grid3x3_loss005.yaml` | V2 with loss coefficient 0.05 | current best result |
| `v2_grid3x3_loss01.yaml` | V2 with loss coefficient 0.1 | intermediate tuning run |
| `v3_grid3x3_uncertainty.yaml` | V2.1 + depth uncertainty weighting | uncertainty-aware extension; uses regional uncertainty loss |
| `v3_grid3x3_uncertainty_soft.yaml` | V3 with softer uncertainty weighting | completed; worse than V3 and baseline |
| `v4_grid3x3_mask.yaml` | V2 + region-mask guidance, loss coefficient 0.2 | first mask run; not fully fair against V2.1 |
| `v4_grid3x3_mask_loss005.yaml` | V4 with loss coefficient 0.05 | completed; better than V4, but below V2.1 |
| `v5_grid3x3_uncertainty_mask.yaml` | V2 + uncertainty + mask | only useful after V3/V4 are checked |
| `v6_adaptive_region_fusion.yaml` | V2.1 + adaptive query-region fusion | completed; below V2.1 |
| `v7_region_reliability_aux.yaml` | V2.1 + auxiliary region reliability head | next reliability experiment |
| `v7_region_reliability_weighted.yaml` | V2.1 + reliability-guided region weighting | V7.2; reliability changes the region aggregation |

All current variants use ROIAlign on the predicted 2D object box. The grid is pooled from inside the object ROI, not globally from the full feature map.

The regional depth correction is multiplied by a learnable gate initialized at zero. This keeps the starting behavior close to the original MonoDGP baseline and lets the model learn whether the regional correction is useful.

V3 adds an uncertainty head for each cell in the ROI grid. Regions with higher predicted uncertainty receive lower aggregation weight. The uncertainty head is also supervised with a small auxiliary loss based on the region-level depth residual error.

`v3_grid3x3_uncertainty_soft.yaml` was tested as a safer follow-up to V3. It keeps uncertainty-based weighting, but removes the auxiliary uncertainty loss and reduces the uncertainty temperature. The result is worse than V3 and below the baseline, so the current uncertainty weighting design should not be used as the main branch.

`v4_grid3x3_mask_loss005.yaml` is the fair mask-guidance comparison. The first V4 run used `region_geometry_loss_coef=0.2`, while the best V2.1 variant used `0.05`. V4.1 keeps the mask idea but uses the same regional loss strength as V2.1. The result improves over V4 and beats the baseline on AP3D Moderate, but remains below V2.1, so mask guidance is not the current best branch.

`v6_adaptive_region_fusion.yaml` keeps the stable parts of V2.1: ROI grid 3x3, `region_geometry_loss_coef=0.05`, and weak depth residual correction. The difference is that the region weights are predicted from both the object query and each local region feature. It also uses a query-dependent depth gate instead of only one global scalar gate. The result is below V2.1, so V6 is treated as an ablation.

`v7_region_reliability_aux.yaml` starts again from the V2.1 path and adds a reliability head for every ROI cell. The first version is auxiliary-only: reliability is supervised, but it does not change the final depth prediction yet. This checks whether the model can learn which local regions are useful for depth correction before using reliability to modify depth.

`v7_region_reliability_weighted.yaml` is the direct follow-up to V7.1. It keeps the same reliability target, but uses the predicted reliability as a bias on the region logits before softmax. In simple terms, reliable ROI cells should receive more weight when the regional depth correction is computed.

For fair comparison, use the same dataset split, the same epoch count, and the same visualization image IDs.
