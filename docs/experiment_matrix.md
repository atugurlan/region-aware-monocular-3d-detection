# Experiment matrix

This file tracks the dissertation implementation stage. The original plan was Baseline + V1-V5. After the first runs, the plan became more focused: stabilize the ROI-grid branch first, then add uncertainty or mask guidance only if the simpler branch is useful.

## Main experiments

| Name | Description | Output folder | Current status |
| --- | --- | --- | --- |
| Baseline | Original MonoDGP, no region-aware correction | `../experiments/runs/baseline_20ep/` | completed |
| V1 | 2x2 ROI-grid depth geometry correction with learnable gate | `../experiments/runs/v1_roi_grid2x2_gated/` | optional/not prioritized |
| V2 | 3x3 ROI-grid depth geometry correction with learnable gate | `../experiments/runs/v2_roi_grid3x3_gated/` | completed, below baseline |
| V2.1 | V2 with regional loss coefficient 0.05 | `../experiments/runs/v2_roi_grid3x3_gated_loss005/` | completed, current best |
| V2.2 | V2 with regional loss coefficient 0.1 | `../experiments/runs/v2_roi_grid3x3_gated_loss01/` | completed, better than baseline but below V2.1 |
| V3 | 3x3 ROI-grid correction with depth uncertainty weighting | `../experiments/runs/v3_roi_grid3x3_uncertainty_gated/` | completed; below V2.1 |
| V3.1 | V3 with softer uncertainty weighting and no uncertainty auxiliary loss | `../experiments/runs/v3_roi_grid3x3_uncertainty_soft_gated/` | completed; worse than V3 and baseline |
| V4 | 3x3 ROI-grid correction with region-mask guidance | `../experiments/runs/v4_roi_grid3x3_mask_gated/` | completed; below baseline |
| V4.1 | V4 with regional loss coefficient 0.05 | `../experiments/runs/v4_roi_grid3x3_mask_gated_loss005/` | completed; better than V4, below V2.1 |
| V5 | 3x3 ROI-grid correction with uncertainty weighting and mask guidance | `../experiments/runs/v5_roi_grid3x3_uncertainty_mask_gated/` | completed; weakest region-aware variant |
| V6 | Adaptive query-region fusion with query-level depth gate | `../experiments/runs/v6_roi_grid3x3_adaptive_region_fusion/` | completed; below baseline and V2.1 |
| V7.1 | Region reliability head with auxiliary-only supervision | `../experiments/runs/v7_roi_grid3x3_region_reliability_aux/` | completed; below baseline and V2.1 |
| V7.2 | Region reliability used as a bias for ROI-cell weighting | `../experiments/runs/v7_roi_grid3x3_region_reliability_weighted/` | completed; better than baseline/V7.1 but below V2.1 |
| V7.3 | Region reliability used as a gate on the final depth delta | `../experiments/runs/v7_roi_grid3x3_region_reliability_delta_gate/` | completed/rerun; below baseline and V7.2 |
| V7.4 | Soft reliability gate on the final depth delta | `../experiments/runs/v7_roi_grid3x3_region_reliability_soft_delta_gate/` | ready to run |

## Fallback experiments

| Name | Description | When to use |
| --- | --- | --- |
| F1 | Auxiliary-only regional residual, no depth injection | if depth correction becomes unstable |
| F2 | Query-dependent gate instead of one scalar gate | if one global gate is too limited |
| F3 | Region-aware uncertainty instead of direct depth correction | if changing depth directly hurts AP3D |
| F4 | Feature-level ROI refinement before the existing depth head | if residual prediction is too shallow |

## Decision rule

V2.1 is currently the strongest branch. It should be used as the reference region-aware model for the next stage.

The current decision is to keep V2.1 as the main dissertation variant. V3, V3.1, V4, V4.1, and V5 are useful ablations, but they do not improve the best ROI-grid correction. V3.1 was tested to check whether the original uncertainty run was hurt by the auxiliary uncertainty loss. It performed worse than V3 and worse than the baseline, so the current uncertainty design should not be used as the main direction. V4.1 improves over V4 but remains below V2.1, so mask guidance is also not the best current path.

The next useful paths are:

1. Keep V2.1 as the main experimental result for the first stage.
2. Use V3, V3.1, V4, V4.1, and V5 as ablations that explain what did not help.
3. Treat V6 as evidence that a more expressive fusion block alone does not solve the depth correction problem.
4. Keep V2.1 as the stable result for the first stage.
5. Treat V7.1 as evidence that auxiliary reliability alone does not improve AP3D in the current design.
6. Treat V7.2 as evidence that reliability is useful only when it affects aggregation, not only as auxiliary supervision.
7. Treat V7.3 as evidence that reliability should not simply multiply the final depth delta. The rerun confirms this, with AP3D Moderate 11.1904 at best epoch 19.
8. Keep V7.2 as the useful reliability ablation, but keep V2.1 as the strongest completed result.
9. Run V7.4 next if we want to check whether a softer delta gate fixes V7.3 without jumping to V6.1.
10. If V7.4 also fails, prefer V6.1 adaptive residual logits over another delta-gating variant.

## Implementation note

The current variants use ROIAlign on the predicted 2D object box for each object query. This is different from the first historical attempt, where the grid was pooled globally from the feature map. The ROI version is closer to the dissertation idea because the regions are actually inside the estimated object area.

The regional depth correction is multiplied by a learnable gate initialized at zero. This avoids forcing a weak regional head into the depth prediction too early in training.

V6 changes this gate and aggregation strategy. Instead of using only query-level logits over the region grid and one global scalar gate, V6 predicts region weights from the interaction between each object query and each ROI cell. It also predicts a gate per query, so different detected objects can use different amounts of regional correction. The 20-epoch result did not improve AP3D over V2.1, so this design is useful as an ablation rather than as the final candidate.

V7.1 returns to the simpler V2.1 correction path and adds a reliability head per ROI cell. Reliability is supervised with a target derived from the error between each cell residual and the matched object-level depth residual. The first V7 version is auxiliary-only, so it should not destabilize final depth unless the extra supervision changes the shared region features too much. The 20-epoch result is below V2.1, which means this auxiliary signal is not enough by itself.

V7.2 keeps the same reliability supervision, but uses the reliability prediction inside the ROI-cell aggregation. The reliability score is added as a bias to the region logits before softmax. This means a region that looks more reliable can receive a larger weight when the final regional depth correction is computed. The result improves clearly over V7.1 and also beats the baseline, but it remains below V2.1. The next reliability step should change how reliability gates the final depth delta, not only how it changes the softmax over ROI cells.

V7.3 keeps the ROI-cell aggregation closer to V2.1 and uses reliability after aggregation. The weighted reliability score becomes an extra multiplicative gate on the final regional depth delta. The rerun reached best epoch 19 with AP3D Moderate 11.1904. This is still below the baseline and below V7.2, so the gate is probably too restrictive. The reliability signal is more useful when it affects region aggregation, as in V7.2, than when it directly shrinks the final depth correction. V7.3 should be kept as a negative ablation, not as a candidate for longer training.

V7.4 changes the V7.3 gate into a centered soft scale. Instead of forcing the depth delta to be multiplied by a value in `[0, 1]`, the soft gate is `1 + alpha * (reliability - 0.5)`. This keeps the correction near the V2.1 behavior while still allowing reliability to reduce or increase it slightly.

The most important comparison is not only whether AP improves. The experiments should also show how the model changes failure cases. A useful variant should improve AP3D without making projected 3D boxes visibly worse on common KITTI examples.
