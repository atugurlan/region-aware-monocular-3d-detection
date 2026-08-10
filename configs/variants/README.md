# Variant configs

This folder contains runnable configs for the first ablation stage.

The experiment order is:

1. `v1_grid2x2.yaml`
2. `v2_grid3x3.yaml`
3. `v3_grid3x3_uncertainty.yaml`
4. `v4_grid3x3_mask.yaml`
5. `v5_grid3x3_uncertainty_mask.yaml`

These are not final dissertation variants. They are ablation runs used to understand which region-aware idea is useful.

All five variants use ROIAlign on each predicted 2D object box. The grid is pooled from inside the object ROI, not globally from the full feature map.

Extra tuning configs may be added when a variant is close to the baseline but needs a smaller change. For example, `v2_grid3x3_loss005.yaml` keeps V2 but reduces the regional geometry loss coefficient from `0.2` to `0.05`. `v2_grid3x3_loss01.yaml` tests the intermediate coefficient `0.1`.

For a quick check, run each config for a small number of epochs first. After that, use the same number of epochs for all V1-V5 comparisons.
