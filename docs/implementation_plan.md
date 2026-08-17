# Implementation plan

This repository starts from MonoDGP and keeps the original model as the baseline. The dissertation direction is region-aware monocular 3D object detection. The plan below is based on the results obtained so far, not only on the initial idea.

## Current conclusion

The first important conclusion is that the regional module should not be forced directly into the depth prediction. Direct ROI correction reduced AP3D. Adding a learnable gate made the branch safer. Reducing the regional geometry loss coefficient gave the first clear improvement over the baseline.

The current best variant is V2.1:

- ROIAlign inside each predicted 2D object box;
- 3x3 regional feature grid;
- regional depth residual head;
- learnable scalar depth gate initialized at zero;
- `region_geometry_loss_coef=0.05`;
- 20 epochs on KITTI.

V2.1 reached AP3D Moderate 14.4722, compared with 12.1207 for the 20-epoch baseline.

## Baseline

The original MonoDGP pipeline contains three useful parts for the proposed work:

- a 2D detection transformer, which builds object queries from image features;
- a depth predictor, which estimates depth-related features;
- a region segmentation head, which produces region probabilities and enhanced image features before the 3D decoder.

MonoDGP also predicts a geometry-based depth correction through `depth_embed`. This is the main insertion point for the current experiments. The baseline keeps this original behavior unchanged.

## What was tried

The first global-grid attempt pooled regions from the full feature map. It gave some numerical improvement, but it was not really region-aware inside the object. It is kept only as implementation history.

The first proper ROI-grid attempt used ROIAlign inside each predicted 2D box and added a regional depth correction directly to `depth_geo_err`. This was conceptually better, but AP3D dropped to 9.3288 Moderate. The branch was probably changing depth too strongly before it learned a reliable residual.

The gated ROI-grid variant added a learnable scalar gate initialized at zero. This helped stability. With loss coefficient 0.2, AP3D Moderate reached 10.9176. With loss coefficient 0.1, AP3D Moderate reached 12.5843. With loss coefficient 0.05, AP3D Moderate reached 14.4722.

## Next implementation steps

The next steps should stay controlled:

1. Keep V2.1 as the current best reference.
2. Parse and visualize V2.2, then compare it with V2.1 and baseline.
3. Decide if V2.1 should be trained for more epochs.
4. If the V2.1 behavior remains stable, implement V3 with uncertainty weighting.
5. If uncertainty helps, test V4 or V5.
6. If uncertainty hurts, keep the dissertation focused on gated ROI regional geometry and analyze it more deeply.

## Fallback architecture options

The first fallback is an auxiliary-only regional branch. The model predicts the regional depth residual and receives `loss_region_geometry`, but the residual is not added to `pred_depth`. This checks whether the branch can learn useful geometry without disturbing the detector.

The second fallback is a query-dependent gate. Instead of one global scalar gate, the model predicts a gate for each object query:

```text
gate = sigmoid(MLP(query_feature))
depth_final = depth_base + gate * region_depth_delta
```

This is more flexible than the current scalar gate because each object can decide how much regional correction it needs.

The third fallback is region-aware uncertainty. Instead of changing the depth value directly, the regional branch modifies depth confidence or log-variance. This may be safer because the model does not move the 3D box directly.

V3 now implements the first version of this idea on top of V2.1. Each ROI grid cell predicts both a depth residual and an uncertainty value. The uncertainty lowers the aggregation weight of unreliable cells, and a small auxiliary uncertainty loss teaches the head to assign larger uncertainty to cells whose residual is far from the matched target residual.

The fourth fallback is feature-level refinement. ROI features are fused back into the query feature, and then the existing depth head predicts depth from the refined query. This is a stronger architecture change, so it should come after the simpler variants are understood.

## Evaluation

Each comparable run should use:

- KITTI 3D object detection validation split;
- 20 epochs for the first comparison;
- AP_R40 metrics;
- Car AP3D and APBEV;
- Easy, Moderate, and Hard difficulty levels;
- the same visualization set for projected 3D boxes.

The main metric is AP3D Moderate. AP3D Hard is also important because regional geometry should help especially with occluded, truncated, or distant objects. APBEV is useful for checking whether localization in bird's-eye view improves together with true 3D detection.

## Dissertation contribution

The contribution should not be described as only reproducing MonoDGP. The contribution is an implementation and analysis of region-aware geometry-error modeling on top of MonoDGP.

The useful dissertation story is the comparison between several ways of injecting regional geometry:

- direct depth correction;
- gated depth correction;
- weaker regional supervision;
- uncertainty-based weighting;
- mask-guided region weighting;
- auxiliary-only regional learning if needed.

The first results already show that the way the regional correction is connected matters a lot. This gives the dissertation a concrete technical angle: not only whether regional information helps, but how it should be introduced without destabilizing monocular depth estimation.
