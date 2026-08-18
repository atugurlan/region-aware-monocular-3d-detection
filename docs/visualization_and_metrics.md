# Visualization and metrics workflow

This file describes the analysis artifacts that should be created after each run.

## 1. Projected 3D boxes

After a run finishes, MonoDGP writes prediction files in KITTI format under the run output folder:

```text
../experiments/runs/<run_name>/<model_name>/outputs/data/
```

Use `tools/visualize_kitti_predictions.py` to draw projected 3D boxes on the original KITTI images.

Example for the baseline:

```powershell
python -B tools/visualize_kitti_predictions.py `
  --data_root ../data/kitti `
  --pred_dir ../experiments/runs/baseline_20ep/monodgp/outputs/data `
  --out_dir ../experiments/visualizations/baseline_20ep `
  --max_images 20
```

Colors:

- green = ground truth 3D box projected on the image;
- red = predicted 3D box projected on the image.

For dissertation analysis, save both good and bad examples. Bad examples are useful because they show occlusion, truncation, depth ambiguity, and wrong orientation.

## 2. Metrics CSV

Training and validation logs can be parsed into CSV with `tools/parse_metrics.py`.

Example:

```powershell
python -B tools/parse_metrics.py `
  --log ../experiments/runs/baseline_20ep/monodgp/train.log.* `
  --out_csv ../experiments/metrics/baseline_20ep_metrics.csv
```

The CSV contains:

- epoch;
- class;
- protocol: `AP` or `AP_R40`;
- overlap thresholds for Easy, Moderate, and Hard;
- metric type: `bbox`, `bev`, `3d`, `aos`;
- Easy, Moderate, Hard values;
- best result and best epoch when available.

For the main dissertation comparison, filter the CSV to `protocol=AP_R40`, overlap thresholds `0.70, 0.70, 0.70`, metric `3d`, and class `Car`.

## 3. Later comparison plots

After V1-V5 are run, the CSV files can be combined to plot:

- AP3D Moderate over epochs for each variant;
- APBEV Moderate over epochs for each variant;
- final AP3D comparison between Baseline and V1-V5;
- final APBEV comparison between Baseline and V1-V5;
- separate Easy / Moderate / Hard comparison.

The visual analysis should use the same selected image IDs for Baseline and V1-V5. This makes the comparison easier to explain in the dissertation.

## 4. Current V4.1 artifacts

V4.1 was visualized after the 20-epoch run with:

```powershell
python -B tools/visualize_kitti_predictions.py `
  --data_root ../data/kitti `
  --pred_dir ../experiments/runs/v4_roi_grid3x3_mask_gated_loss005/v4_roi_grid3x3_mask_gated_loss005/outputs/data `
  --out_dir ../experiments/visualizations/v4_roi_grid3x3_mask_gated_loss005 `
  --max_images 20
```

Generated folders:

- `../experiments/visualizations/v4_roi_grid3x3_mask_gated_loss005/`
- `../experiments/visualizations/comparison_baseline_vs_v4_roi_grid3x3_mask_gated_loss005/`
- `../experiments/visualizations/comparison_v2_loss005_vs_v4_roi_grid3x3_mask_gated_loss005/`
- `../experiments/visualizations/comparison_v4_vs_v4_loss005/`

The V4.1 metric CSV is:

```text
../experiments/metrics/v4_roi_grid3x3_mask_gated_loss005_metrics.csv
```


## 5. Current V3.1 artifacts

V3.1 was visualized after the 20-epoch run with:

```powershell
python -B tools/visualize_kitti_predictions.py `
  --data_root ../data/kitti `
  --pred_dir ../experiments/runs/v3_roi_grid3x3_uncertainty_soft_gated/v3_roi_grid3x3_uncertainty_soft_gated/outputs/data `
  --out_dir ../experiments/visualizations/v3_roi_grid3x3_uncertainty_soft_gated `
  --max_images 20
```

Generated folders:

- `../experiments/visualizations/v3_roi_grid3x3_uncertainty_soft_gated/`
- `../experiments/visualizations/comparison_baseline_vs_v3_1_soft_uncertainty/`
- `../experiments/visualizations/comparison_v2_loss005_vs_v3_1_soft_uncertainty/`
- `../experiments/visualizations/comparison_v3_vs_v3_1_soft_uncertainty/`

The V3.1 metric CSV is:

```text
../experiments/metrics/v3_roi_grid3x3_uncertainty_soft_gated_metrics.csv
```

The generated plots are:

- `../experiments/plots/v3_1_soft_uncertainty_training_curve.png`
- `../experiments/plots/variant_comparison_ap3d_moderate.png`
- `../experiments/plots/variant_comparison_moderate_metrics.png`
- `../experiments/plots/variant_best_metrics_summary.md`


## 6. Current V2.2 artifacts

V2.2 was visualized after the 20-epoch run with:

```powershell
python -B tools/visualize_kitti_predictions.py `
  --data_root ../data/kitti `
  --pred_dir ../experiments/runs/v2_roi_grid3x3_gated_loss01/v2_roi_grid3x3_gated_loss01/outputs/data `
  --out_dir ../experiments/visualizations/v2_roi_grid3x3_gated_loss01 `
  --max_images 20
```

Generated folders:

- `../experiments/visualizations/v2_roi_grid3x3_gated_loss01/`
- `../experiments/visualizations/comparison_baseline_vs_v2_roi_grid3x3_gated_loss01/`
- `../experiments/visualizations/comparison_v2_loss005_vs_v2_loss01/`
