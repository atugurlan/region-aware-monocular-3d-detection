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
| V3 | `configs/variants/v3_grid3x3_uncertainty.yaml` | 20 | completed | 12 | 17.0721 | 12.4507 | 9.6186 | 24.6198 | 17.5752 | 14.2985 | Uncertainty weighting is slightly above baseline on Moderate, but below V2.1. |
| V3.1 | `configs/variants/v3_grid3x3_uncertainty_soft.yaml` | 20 | completed | 19 | 15.9107 | 10.9410 | 8.9183 | 23.6161 | 16.2183 | 13.5886 | Softer uncertainty without auxiliary uncertainty loss is worse than V3 and baseline. |
| V4 | `configs/variants/v4_grid3x3_mask.yaml` | 20 | completed | 17 | 15.9572 | 11.8357 | 9.3153 | 25.1234 | 17.9596 | 14.5451 | Mask guidance is below baseline and below V2.1/V3. |
| V4.1 | `configs/variants/v4_grid3x3_mask_loss005.yaml` | 20 | completed | 19 | 18.7841 | 13.6364 | 10.9500 | 27.3974 | 19.4732 | 16.2057 | Fair mask run with loss 0.05. Better than baseline and V4, but still below V2.1. |
| V5 | `configs/variants/v5_grid3x3_uncertainty_mask.yaml` | 20 | completed | 13 | 13.6815 | 9.9023 | 8.0324 | 20.6197 | 14.9392 | 12.5606 | Combining uncertainty and mask is the weakest region-aware variant. |
| V6 | `configs/variants/v6_adaptive_region_fusion.yaml` | 20 | completed | 20 | 14.1765 | 11.5201 | 9.4688 | 22.5077 | 17.6326 | 14.8261 | Adaptive fusion improves the design, but not AP3D. It is below baseline and V2.1. |
| V7.1 | `configs/variants/v7_region_reliability_aux.yaml` | 20 | completed | 19 | 14.3914 | 10.7080 | 8.6103 | 22.6127 | 16.5019 | 13.4920 | Reliability auxiliary-only is below baseline and V2.1. |

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

Artifacts:

- metrics CSV: `../experiments/metrics/v2_roi_grid3x3_gated_loss01_metrics.csv`;
- projected 3D boxes: `../experiments/visualizations/v2_roi_grid3x3_gated_loss01/`;
- baseline comparison images: `../experiments/visualizations/comparison_baseline_vs_v2_roi_grid3x3_gated_loss01/`;
- V2.1 comparison images: `../experiments/visualizations/comparison_v2_loss005_vs_v2_loss01/`.

## V3 ROI 3x3 + uncertainty weighting notes

Config: `configs/variants/v3_grid3x3_uncertainty.yaml`

Output folder: `../experiments/runs/v3_roi_grid3x3_uncertainty_gated/`

Best result:

- best epoch: 12;
- 3D AP: Easy 17.0721, Moderate 12.4507, Hard 9.6186;
- BEV AP: Easy 24.6198, Moderate 17.5752, Hard 14.2985;
- 2D bbox AP: Easy 69.7210, Moderate 61.0293, Hard 54.3199.

V3 is slightly better than the baseline on AP3D Moderate, but it is clearly below V2.1. This means the uncertainty weighting does not currently improve the best region-aware branch. The result is still useful as an ablation because it shows that adding uncertainty on top of the ROI correction is not automatically beneficial.

Artifacts:

- metrics CSV: `../experiments/metrics/v3_roi_grid3x3_uncertainty_gated_metrics.csv`;
- projected 3D boxes: `../experiments/visualizations/v3_roi_grid3x3_uncertainty_gated/`;
- baseline comparison images: `../experiments/visualizations/comparison_baseline_vs_v3_roi_grid3x3_uncertainty_gated/`;
- V2.1 comparison images: `../experiments/visualizations/comparison_v2_loss005_vs_v3_roi_grid3x3_uncertainty_gated/`.

## V3.1 ROI 3x3 + soft uncertainty weighting notes

Config: `configs/variants/v3_grid3x3_uncertainty_soft.yaml`

Output folder: `../experiments/runs/v3_roi_grid3x3_uncertainty_soft_gated/`

Best result:

- best epoch: 19;
- 3D AP: Easy 15.9107, Moderate 10.9410, Hard 8.9183;
- BEV AP: Easy 23.6161, Moderate 16.2183, Hard 13.5886;
- 2D bbox AP: Easy 73.5348, Moderate 67.8693, Hard 61.0669.

V3.1 removed the auxiliary uncertainty loss and reduced the uncertainty temperature to 0.5. The goal was to check whether V3 was hurt by uncertainty supervision being too strong. The result is worse than V3 and also below the baseline on AP3D Moderate. This suggests that the current uncertainty branch is not only over-supervised; the weighting mechanism itself may be suppressing useful regional evidence. For the next stage, uncertainty should not be added as a simple penalty on region logits. A better direction is to make the fusion query-dependent or to predict region reliability in a way that is evaluated separately from direct depth correction.

Artifacts:

- metrics CSV: `../experiments/metrics/v3_roi_grid3x3_uncertainty_soft_gated_metrics.csv`;
- projected 3D boxes: `../experiments/visualizations/v3_roi_grid3x3_uncertainty_soft_gated/`;
- baseline comparison images: `../experiments/visualizations/comparison_baseline_vs_v3_1_soft_uncertainty/`;
- V2.1 comparison images: `../experiments/visualizations/comparison_v2_loss005_vs_v3_1_soft_uncertainty/`;
- V3 comparison images: `../experiments/visualizations/comparison_v3_vs_v3_1_soft_uncertainty/`.

## V4 ROI 3x3 + mask guidance notes

Config: `configs/variants/v4_grid3x3_mask.yaml`

Output folder: `../experiments/runs/v4_roi_grid3x3_mask_gated/`

Best result:

- best epoch: 17;
- 3D AP: Easy 15.9572, Moderate 11.8357, Hard 9.3153;
- BEV AP: Easy 25.1234, Moderate 17.9596, Hard 14.5451;
- 2D bbox AP: Easy 86.0895, Moderate 75.6151, Hard 67.8422.

V4 is weaker than the baseline on AP3D Moderate and clearly weaker than V2.1. It also does not improve over V3. The mask guidance may be too restrictive, or the region mask may remove useful ROI evidence before the regional depth residual is estimated. For the dissertation, this is a useful negative ablation because it suggests that simply masking ROI features is not enough.

Artifacts:

- metrics CSV: `../experiments/metrics/v4_roi_grid3x3_mask_gated_metrics.csv`;
- projected 3D boxes: `../experiments/visualizations/v4_roi_grid3x3_mask_gated/`;
- baseline comparison images: `../experiments/visualizations/comparison_baseline_vs_v4_roi_grid3x3_mask_gated/`;
- V2.1 comparison images: `../experiments/visualizations/comparison_v2_loss005_vs_v4_roi_grid3x3_mask_gated/`;
- V3 comparison images: `../experiments/visualizations/comparison_v3_vs_v4_roi_grid3x3_mask_gated/`.

## V4.1 ROI 3x3 + mask guidance, loss 0.05 notes

Config: `configs/variants/v4_grid3x3_mask_loss005.yaml`

Output folder: `../experiments/runs/v4_roi_grid3x3_mask_gated_loss005/`

Best result:

- best epoch: 19;
- 3D AP: Easy 18.7841, Moderate 13.6364, Hard 10.9500;
- BEV AP: Easy 27.3974, Moderate 19.4732, Hard 16.2057;
- 2D bbox AP: Easy 81.6054, Moderate 73.6325, Hard 66.9220.

V4.1 is the fair mask comparison against V2.1 because both use `region_geometry_loss_coef=0.05`. It improves clearly over the first V4 run and also beats the baseline on AP3D Moderate. However, it remains below V2.1. This suggests that the stronger loss was part of the V4 problem, but the mask itself still does not improve the simpler ROI-grid correction. The likely explanation is that masking removes some useful context from the ROI features.

Artifacts:

- metrics CSV: `../experiments/metrics/v4_roi_grid3x3_mask_gated_loss005_metrics.csv`;
- projected 3D boxes: `../experiments/visualizations/v4_roi_grid3x3_mask_gated_loss005/`;
- baseline comparison images: `../experiments/visualizations/comparison_baseline_vs_v4_roi_grid3x3_mask_gated_loss005/`;
- V2.1 comparison images: `../experiments/visualizations/comparison_v2_loss005_vs_v4_roi_grid3x3_mask_gated_loss005/`;
- V4 comparison images: `../experiments/visualizations/comparison_v4_vs_v4_loss005/`.

## V5 ROI 3x3 + uncertainty weighting + mask guidance notes

Config: `configs/variants/v5_grid3x3_uncertainty_mask.yaml`

Output folder: `../experiments/runs/v5_roi_grid3x3_uncertainty_mask_gated/`

Best result:

- best epoch: 13;
- 3D AP: Easy 13.6815, Moderate 9.9023, Hard 8.0324;
- BEV AP: Easy 20.6197, Moderate 14.9392, Hard 12.5606;
- 2D bbox AP: Easy 78.3467, Moderate 68.5798, Hard 63.8607.

V5 is the weakest of the tested region-aware variants. It is below the baseline and far below V2.1. Since V3 and V4 were already weaker than V2.1 separately, this result confirms that combining uncertainty weighting and mask guidance does not help in the current design. The likely issue is that the model receives two filtering mechanisms at once: uncertain cells are downweighted, and masked features are already reduced before the regional residual is estimated.

Artifacts:

- metrics CSV: `../experiments/metrics/v5_roi_grid3x3_uncertainty_mask_gated_metrics.csv`;
- projected 3D boxes: `../experiments/visualizations/v5_roi_grid3x3_uncertainty_mask_gated/`;
- baseline comparison images: `../experiments/visualizations/comparison_baseline_vs_v5_roi_grid3x3_uncertainty_mask_gated/`;
- V2.1 comparison images: `../experiments/visualizations/comparison_v2_loss005_vs_v5_roi_grid3x3_uncertainty_mask_gated/`;
- V3 comparison images: `../experiments/visualizations/comparison_v3_vs_v5_roi_grid3x3_uncertainty_mask_gated/`;
- V4 comparison images: `../experiments/visualizations/comparison_v4_vs_v5_roi_grid3x3_uncertainty_mask_gated/`.

## V6 ROI 3x3 + adaptive query-region fusion notes

Config: `configs/variants/v6_adaptive_region_fusion.yaml`

Output folder: `../experiments/runs/v6_roi_grid3x3_adaptive_region_fusion/`

Best result:

- best epoch: 20;
- 3D AP: Easy 14.1765, Moderate 11.5201, Hard 9.4688;
- BEV AP: Easy 22.5077, Moderate 17.6326, Hard 14.8261;
- 2D bbox AP: Easy 84.2021, Moderate 77.1912, Hard 70.3618.

V6 tested whether the ROI-grid correction becomes stronger if the model learns region weights from the interaction between the object query and each local ROI cell. It also replaced the single scalar gate with a query-level depth gate. The result is not better than V2.1 and is slightly below the baseline on AP3D Moderate. The 2D box metric is strong, especially on Moderate and Hard, but the 3D depth correction does not benefit enough from the more expressive fusion. This suggests that the current problem is not only how regions are aggregated. The next design should focus more directly on whether each region is reliable for depth correction, or on using the regional branch as an auxiliary signal instead of always injecting a depth residual.

Artifacts:

- metrics CSV: `../experiments/metrics/v6_roi_grid3x3_adaptive_region_fusion_metrics.csv`;
- projected 3D boxes: `../experiments/visualizations/v6_roi_grid3x3_adaptive_region_fusion/`;
- baseline comparison images: `../experiments/visualizations/comparison_baseline_vs_v6_adaptive_region_fusion/`;
- V2.1 comparison images: `../experiments/visualizations/comparison_v2_loss005_vs_v6_adaptive_region_fusion/`;
- V4.1 comparison images: `../experiments/visualizations/comparison_v4_loss005_vs_v6_adaptive_region_fusion/`;
- training curve: `../experiments/plots/v6_adaptive_region_fusion_training_curve.png`.

## V7.1 ROI 3x3 + region reliability auxiliary-only notes

Config: `configs/variants/v7_region_reliability_aux.yaml`

Output folder: `../experiments/runs/v7_roi_grid3x3_region_reliability_aux/`

Best result:

- best epoch: 19;
- 3D AP: Easy 14.3914, Moderate 10.7080, Hard 8.6103;
- BEV AP: Easy 22.6127, Moderate 16.5019, Hard 13.4920;
- 2D bbox AP: Easy 81.1360, Moderate 72.4465, Hard 66.3892.

V7.1 added a region reliability head per ROI cell and trained it with an auxiliary target derived from the local residual error. The reliability output was not used to change final depth. This was intentionally safer than V7.2, because previous variants showed that direct depth modification can reduce AP3D. The result is below the baseline and below V2.1, so auxiliary reliability alone is not enough in the current form. It is still useful as an ablation because it shows that simply asking the model to predict reliable regions does not automatically improve the shared region features.

Artifacts:

- metrics CSV: `../experiments/metrics/v7_roi_grid3x3_region_reliability_aux_metrics.csv`;
- projected 3D boxes: `../experiments/visualizations/v7_roi_grid3x3_region_reliability_aux/`;
- baseline comparison images: `../experiments/visualizations/comparison_baseline_vs_v7_1_region_reliability_aux/`;
- V2.1 comparison images: `../experiments/visualizations/comparison_v2_loss005_vs_v7_1_region_reliability_aux/`;
- V6 comparison images: `../experiments/visualizations/comparison_v6_vs_v7_1_region_reliability_aux/`;
- training curve: `../experiments/plots/v7_1_region_reliability_aux_training_curve.png`.

## Analysis artifacts per run

For each completed run, save these artifacts:

| Run | Metrics CSV | Projected 3D boxes | Comparison images |
| --- | --- | --- | --- |
| Baseline | `../experiments/metrics/baseline_20ep_metrics.csv` | `../experiments/visualizations/baseline_20ep/` | used as reference |
| V2.1 | `../experiments/metrics/v2_roi_grid3x3_gated_loss005_metrics.csv` | `../experiments/visualizations/v2_roi_grid3x3_gated_loss005/` | `../experiments/visualizations/comparison_baseline_vs_v2_roi_grid3x3_gated_loss005/` |
| V2.2 | `../experiments/metrics/v2_roi_grid3x3_gated_loss01_metrics.csv` | `../experiments/visualizations/v2_roi_grid3x3_gated_loss01/` | `../experiments/visualizations/comparison_baseline_vs_v2_roi_grid3x3_gated_loss01/`, `../experiments/visualizations/comparison_v2_loss005_vs_v2_loss01/` |
| V3 | `../experiments/metrics/v3_roi_grid3x3_uncertainty_gated_metrics.csv` | `../experiments/visualizations/v3_roi_grid3x3_uncertainty_gated/` | `../experiments/visualizations/comparison_baseline_vs_v3_roi_grid3x3_uncertainty_gated/`, `../experiments/visualizations/comparison_v2_loss005_vs_v3_roi_grid3x3_uncertainty_gated/` |
| V4 | `../experiments/metrics/v4_roi_grid3x3_mask_gated_metrics.csv` | `../experiments/visualizations/v4_roi_grid3x3_mask_gated/` | `../experiments/visualizations/comparison_baseline_vs_v4_roi_grid3x3_mask_gated/`, `../experiments/visualizations/comparison_v2_loss005_vs_v4_roi_grid3x3_mask_gated/`, `../experiments/visualizations/comparison_v3_vs_v4_roi_grid3x3_mask_gated/` |
| V3.1 | `../experiments/metrics/v3_roi_grid3x3_uncertainty_soft_gated_metrics.csv` | `../experiments/visualizations/v3_roi_grid3x3_uncertainty_soft_gated/` | `../experiments/visualizations/comparison_baseline_vs_v3_1_soft_uncertainty/`, `../experiments/visualizations/comparison_v2_loss005_vs_v3_1_soft_uncertainty/`, `../experiments/visualizations/comparison_v3_vs_v3_1_soft_uncertainty/` |
| V4.1 | `../experiments/metrics/v4_roi_grid3x3_mask_gated_loss005_metrics.csv` | `../experiments/visualizations/v4_roi_grid3x3_mask_gated_loss005/` | `../experiments/visualizations/comparison_baseline_vs_v4_roi_grid3x3_mask_gated_loss005/`, `../experiments/visualizations/comparison_v2_loss005_vs_v4_roi_grid3x3_mask_gated_loss005/`, `../experiments/visualizations/comparison_v4_vs_v4_loss005/` |
| V5 | `../experiments/metrics/v5_roi_grid3x3_uncertainty_mask_gated_metrics.csv` | `../experiments/visualizations/v5_roi_grid3x3_uncertainty_mask_gated/` | `../experiments/visualizations/comparison_baseline_vs_v5_roi_grid3x3_uncertainty_mask_gated/`, `../experiments/visualizations/comparison_v2_loss005_vs_v5_roi_grid3x3_uncertainty_mask_gated/`, `../experiments/visualizations/comparison_v3_vs_v5_roi_grid3x3_uncertainty_mask_gated/`, `../experiments/visualizations/comparison_v4_vs_v5_roi_grid3x3_uncertainty_mask_gated/` |
| V6 | `../experiments/metrics/v6_roi_grid3x3_adaptive_region_fusion_metrics.csv` | `../experiments/visualizations/v6_roi_grid3x3_adaptive_region_fusion/` | `../experiments/visualizations/comparison_baseline_vs_v6_adaptive_region_fusion/`, `../experiments/visualizations/comparison_v2_loss005_vs_v6_adaptive_region_fusion/`, `../experiments/visualizations/comparison_v4_loss005_vs_v6_adaptive_region_fusion/` |
| V7.1 | `../experiments/metrics/v7_roi_grid3x3_region_reliability_aux_metrics.csv` | `../experiments/visualizations/v7_roi_grid3x3_region_reliability_aux/` | `../experiments/visualizations/comparison_baseline_vs_v7_1_region_reliability_aux/`, `../experiments/visualizations/comparison_v2_loss005_vs_v7_1_region_reliability_aux/`, `../experiments/visualizations/comparison_v6_vs_v7_1_region_reliability_aux/` |

## Next steps

1. Keep V2.1 as the current reference region-aware variant.
2. Treat V3.1 as evidence that softer uncertainty weighting does not fix the uncertainty branch.
3. Treat V4.1 as evidence that weaker regional supervision helps mask guidance, but that mask guidance is still below V2.1.
4. Treat V6 as evidence that adaptive query-region fusion alone is not enough to improve AP3D.
5. Treat V7.1 as an auxiliary reliability ablation. V2.1 remains the strongest completed result. The next step should be either V7.2 reliability-weighted residual or V8 region-token transformer.

## Dissertation interpretation

The early experiments show a useful pattern. A regional correction can help monocular 3D detection, but only when it is controlled. Directly changing depth from ROI features hurts the model. A learnable gate improves stability. A smaller regional loss coefficient gives the best result so far.

This supports the dissertation idea because the contribution is not just adding a module. The contribution is the analysis of how region-level geometry should be connected to a strong monocular 3D detector.
