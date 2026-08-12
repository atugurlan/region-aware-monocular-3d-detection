# Experiment results log

This document tracks the baseline and region-aware experiments. It should be updated after every run, even when the result is worse than expected. The goal is to keep a clear history for the dissertation implementation chapter.

## Comparison rules

- Use the KITTI validation split for all comparable runs.
- Compare runs with the same number of epochs whenever possible.
- Keep the baseline visible at the top.
- Record failed or weaker runs too, because they explain why the architecture changed.
- Use AP_R40 for the main comparison.
- The main metric is Car AP3D Moderate at IoU 0.70.

## Experiment summary table

| Run | Config | Epochs | Status | Best epoch | AP3D Easy | AP3D Mod. | AP3D Hard | APBEV Easy | APBEV Mod. | APBEV Hard | Notes |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | `configs/monodgp_baseline_20.yaml` | 20 | completed | 20 | 16.6493 | 12.1207 | 9.6970 | 23.4640 | 17.4990 | 14.5575 | Original MonoDGP short baseline. |
| V1 global-grid history | `configs/variants/v1_grid2x2.yaml` old implementation | 20 | completed | TBD | TBD | 12.4054 | TBD | TBD | TBD | TBD | Historical result only; grid was not inside object ROI. |
| V2 global-grid history | `configs/variants/v2_grid3x3.yaml` old implementation | 20 | completed | 20 | TBD | 12.8214 | TBD | TBD | TBD | TBD | Historical result only; not used as final evidence. |
| V2 direct ROI | internal ROI-grid attempt | 20 | completed | 13 | TBD | 9.3288 | TBD | TBD | TBD | TBD | Directly adding regional depth residual hurt AP3D. |
| V2 gated, loss 0.2 | `configs/variants/v2_grid3x3.yaml` | 20 | completed | 19 | 14.0715 | 10.9176 | 9.1391 | 55.6646 | 43.3007 | 38.0107 | Gate helped compared with direct ROI, but still below baseline. |
| V2.1 gated, loss 0.05 | `configs/variants/v2_grid3x3_loss005.yaml` | 20 | completed | 20 | 19.8902 | 14.4722 | 11.7857 | 27.9416 | 19.9727 | 17.0827 | Current best variant. Improves AP3D Moderate by +2.3515 over baseline. |
| V2.2 gated, loss 0.1 | `configs/variants/v2_grid3x3_loss01.yaml` | 20 | completed | 20 | 16.3276 | 12.5843 | 10.3751 | 24.7966 | 18.1860 | 15.4995 | Better than baseline, but weaker than V2.1. |
| V3 | `configs/variants/v3_grid3x3_uncertainty.yaml` | TBD | not started | TBD | TBD | TBD | TBD | TBD | TBD | TBD | ROI 3x3 + uncertainty weighting. |
| V4 | `configs/variants/v4_grid3x3_mask.yaml` | TBD | not started | TBD | TBD | TBD | TBD | TBD | TBD | TBD | ROI 3x3 + region-mask guidance. |
| V5 | `configs/variants/v5_grid3x3_uncertainty_mask.yaml` | TBD | not started | TBD | TBD | TBD | TBD | TBD | TBD | TBD | ROI 3x3 + uncertainty + mask. |

## Current best result

The current best region-aware model is V2.1, which uses:

- ROIAlign inside each predicted 2D object box;
- a 3x3 regional grid;
- regional depth residual prediction;
- learnable scalar depth gate initialized at zero;
- `region_geometry_loss_coef=0.05`;
- `output_scale=0.1`.

Compared with the 20-epoch baseline, V2.1 improves:

| Metric | Baseline | V2.1 | Difference |
| --- | ---: | ---: | ---: |
| AP3D Easy | 16.6493 | 19.8902 | +3.2409 |
| AP3D Moderate | 12.1207 | 14.4722 | +2.3515 |
| AP3D Hard | 9.6970 | 11.7857 | +2.0887 |
| APBEV Easy | 23.4640 | 27.9416 | +4.4776 |
| APBEV Moderate | 17.4990 | 19.9727 | +2.4737 |
| APBEV Hard | 14.5575 | 17.0827 | +2.5252 |

This is useful because the improvement appears on Easy, Moderate, and Hard, not only on one difficulty level. The gain on Hard is also important for the dissertation because harder cases usually contain more occlusion, truncation, or depth ambiguity.

## Baseline notes

Config: `configs/monodgp_baseline_20.yaml`

Output folder: `../experiments/runs/baseline_20ep/`

Best result from the 20-epoch run:

- best epoch: 20;
- AP_R40, Car, IoU 0.70;
- 3D AP: Easy 16.6493, Moderate 12.1207, Hard 9.6970;
- BEV AP: Easy 23.4640, Moderate 17.4990, Hard 14.5575;
- 2D bbox AP: Easy 79.9042, Moderate 72.1897, Hard 67.1772.

This is a short training baseline, not a full paper-level reproduction. It is still valid for the first ablation stage because the variants use the same training budget.

## V2 direct ROI notes

This was the first proper ROI version. It pooled a 3x3 grid from inside each predicted object box and predicted a depth residual. The residual was added directly to the original MonoDGP depth prediction.

The result dropped to AP3D Moderate 9.3288. This showed that the regional branch can damage depth if it is allowed to change the prediction too strongly from the beginning.

## V2 gated, loss 0.2 notes

Config: `configs/variants/v2_grid3x3.yaml`

Output folder: `../experiments/runs/v2_roi_grid3x3_gated/`

Best result:

- best epoch: 19;
- 3D AP: Easy 14.0715, Moderate 10.9176, Hard 9.1391;
- BEV AP: Easy 55.6646, Moderate 43.3007, Hard 38.0107.

The learnable gate made the ROI branch more stable than direct correction. However, AP3D Moderate was still below the baseline, so the regional loss was probably too strong or the correction was still too noisy.

## V2.1 gated, loss 0.05 notes

Config: `configs/variants/v2_grid3x3_loss005.yaml`

Output folder: `../experiments/runs/v2_roi_grid3x3_gated_loss005/`

Best result:

- best epoch: 20;
- 3D AP: Easy 19.8902, Moderate 14.4722, Hard 11.7857;
- BEV AP: Easy 27.9416, Moderate 19.9727, Hard 17.0827;
- 2D bbox AP: Easy 86.0759, Moderate 76.2380, Hard 69.5849.

This is the best result so far. Lowering the regional loss coefficient from 0.2 to 0.05 made the branch less intrusive. The model kept the regional correction useful without letting it dominate the original depth prediction.

Artifacts:

- metrics CSV: `../experiments/metrics/v2_roi_grid3x3_gated_loss005_metrics.csv`;
- projected 3D boxes: `../experiments/visualizations/v2_roi_grid3x3_gated_loss005/`;
- baseline comparison images: `../experiments/visualizations/comparison_baseline_vs_v2_roi_grid3x3_gated_loss005/`.

## V2.2 gated, loss 0.1 notes

Config: `configs/variants/v2_grid3x3_loss01.yaml`

Output folder: `../experiments/runs/v2_roi_grid3x3_gated_loss01/`

Best result:

- best epoch: 20;
- 3D AP: Easy 16.3276, Moderate 12.5843, Hard 10.3751;
- BEV AP: Easy 24.7966, Moderate 18.1860, Hard 15.4995;
- 2D bbox AP: Easy 86.5552, Moderate 79.7132, Hard 73.1259.

This run is better than the baseline on AP3D Moderate and Hard, but it is clearly weaker than V2.1. The result supports the idea that the regional branch is useful, but it needs weak supervision. A coefficient of 0.05 is currently better than 0.1 and 0.2.

## Analysis artifacts per run

For each completed run, save these artifacts:

| Run | Metrics CSV | Projected 3D boxes | Comparison images |
| --- | --- | --- | --- |
| Baseline | `../experiments/metrics/baseline_20ep_metrics.csv` | `../experiments/visualizations/baseline_20ep/` | used as reference |
| V2.1 | `../experiments/metrics/v2_roi_grid3x3_gated_loss005_metrics.csv` | `../experiments/visualizations/v2_roi_grid3x3_gated_loss005/` | `../experiments/visualizations/comparison_baseline_vs_v2_roi_grid3x3_gated_loss005/` |
| V2.2 | `../experiments/metrics/v2_roi_grid3x3_gated_loss01_metrics.csv` | `../experiments/visualizations/v2_roi_grid3x3_gated_loss01/` | TBD |
| V3 | `../experiments/metrics/v3_roi_grid3x3_uncertainty_gated_metrics.csv` | `../experiments/visualizations/v3_roi_grid3x3_uncertainty_gated/` | TBD |
| V4 | `../experiments/metrics/v4_roi_grid3x3_mask_gated_metrics.csv` | `../experiments/visualizations/v4_roi_grid3x3_mask_gated/` | TBD |
| V5 | `../experiments/metrics/v5_roi_grid3x3_uncertainty_mask_gated_metrics.csv` | `../experiments/visualizations/v5_roi_grid3x3_uncertainty_mask_gated/` | TBD |

## Next steps

1. Parse the V2.2 log into a metrics CSV if it has not been parsed yet.
2. Generate projected 3D box visualizations for V2.2.
3. Compare baseline, V2.1, and V2.2 using the same image IDs.
4. Keep V2.1 as the current reference variant.
5. Decide whether to run V3 uncertainty weighting next or first run V2.1 for more epochs.

## Dissertation interpretation

The early experiments show a useful pattern. A regional correction can help monocular 3D detection, but only when it is controlled. Directly changing depth from ROI features hurts the model. A learnable gate improves stability. A smaller regional loss coefficient gives the best result so far.

This supports the dissertation idea because the contribution is not just adding a module. The contribution is the analysis of how region-level geometry should be connected to a strong monocular 3D detector.
