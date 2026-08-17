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
| V3.1 | V3 with softer uncertainty weighting and no uncertainty auxiliary loss | `../experiments/runs/v3_roi_grid3x3_uncertainty_soft_gated/` | ready to run |
| V4 | 3x3 ROI-grid correction with region-mask guidance | `../experiments/runs/v4_roi_grid3x3_mask_gated/` | completed; below baseline |
| V4.1 | V4 with regional loss coefficient 0.05 | `../experiments/runs/v4_roi_grid3x3_mask_gated_loss005/` | ready to run |
| V5 | 3x3 ROI-grid correction with uncertainty weighting and mask guidance | `../experiments/runs/v5_roi_grid3x3_uncertainty_mask_gated/` | completed; weakest region-aware variant |

## Fallback experiments

| Name | Description | When to use |
| --- | --- | --- |
| F1 | Auxiliary-only regional residual, no depth injection | if depth correction becomes unstable |
| F2 | Query-dependent gate instead of one scalar gate | if one global gate is too limited |
| F3 | Region-aware uncertainty instead of direct depth correction | if changing depth directly hurts AP3D |
| F4 | Feature-level ROI refinement before the existing depth head | if residual prediction is too shallow |

## Decision rule

V2.1 is currently the strongest branch. It should be used as the reference region-aware model for the next stage.

The current decision is to keep V2.1 as the main dissertation variant. V3, V4, and V5 are useful ablations, but they do not improve the best ROI-grid correction. Before fully rejecting uncertainty and mask guidance, two fairer follow-up configs should be checked: V3.1 and V4.1.

The next useful paths are:

1. Run V4.1 first, because original V4 used a stronger regional loss than V2.1.
2. Run V3.1 if there is time, because original V3 may have been hurt by the auxiliary uncertainty loss.
3. If V3.1 and V4.1 remain below V2.1, stop adding modules and train/analyze V2.1 more deeply.

## Implementation note

The current variants use ROIAlign on the predicted 2D object box for each object query. This is different from the first historical attempt, where the grid was pooled globally from the feature map. The ROI version is closer to the dissertation idea because the regions are actually inside the estimated object area.

The regional depth correction is multiplied by a learnable gate initialized at zero. This avoids forcing a weak regional head into the depth prediction too early in training.

The most important comparison is not only whether AP improves. The experiments should also show how the model changes failure cases. A useful variant should improve AP3D without making projected 3D boxes visibly worse on common KITTI examples.
