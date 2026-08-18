# Region-Aware Monocular 3D Detection

This repository contains the implementation work for a dissertation project in monocular 3D object detection. The project starts from [MonoDGP](https://github.com/PuFanqi23/MonoDGP), which is used as the baseline model.

The current dissertation direction is:

> Region-aware geometry-error estimation for monocular 3D object detection.

## Motivation

Monocular 3D object detection is difficult because the model has to estimate 3D location, size, orientation, and depth from a single RGB image. Depth is the most fragile part, especially when the object is far away, occluded, truncated, or only partly visible.

MonoDGP already models geometry error at object/query level. This project studies whether the geometry-error signal can be made more local. Instead of using only one global object representation, the model also looks at small regions inside the predicted 2D object box.

The main hypothesis is that not all regions of the same object are equally useful for depth estimation. Some regions may contain stronger geometric cues, while others may be affected by background, occlusion, truncation, or uncertain depth.

## Current Implementation

The current branch adds a region-aware geometry module on top of MonoDGP. For each object query, the predicted 2D box is used as an ROI. ROIAlign extracts a small grid of features from inside that object region. The regional features are then used to predict a depth residual.

The first direct version was too aggressive because the residual was added directly to the depth prediction. The current implementation uses a learnable scalar gate initialized at zero. At the beginning of training, the model behaves like the MonoDGP baseline. During training, it can learn how much regional correction should be used.

The regional depth branch is supervised as a residual between the target depth and the base MonoDGP depth prediction. This keeps the contribution focused on correcting depth, not replacing the whole detector.

## Current Experimental Status

The experiments are tracked in [docs/experiment_results.md](docs/experiment_results.md). The most important result so far is that a smaller regional loss coefficient works better than the initial value.

| Run | Description | Best AP3D Moderate | Status |
| --- | --- | ---: | --- |
| Baseline | Original MonoDGP, 20 epochs | 12.1207 | completed |
| V2 direct ROI | ROI 3x3 correction without gate | 9.3288 | completed, worse than baseline |
| V2 gated, loss 0.2 | ROI 3x3 correction with learnable gate | 10.9176 | completed, still below baseline |
| V2.1 gated, loss 0.05 | Same architecture with weaker regional loss | 14.4722 | completed, current best |
| V2.2 gated, loss 0.1 | Intermediate regional loss | 12.5843 | completed, better than baseline but worse than V2.1 |
| V3 uncertainty | ROI 3x3 with uncertainty weighting | 12.4507 | completed, below V2.1 |
| V3.1 soft uncertainty | V3 without auxiliary uncertainty loss | 10.9410 | completed, worse than V3 and baseline |
| V4.1 mask, loss 0.05 | Fair mask-guidance comparison | 13.6364 | completed, better than V4/baseline but worse than V2.1 |
| V6 adaptive fusion | Query-dependent region fusion and query-level gate | 11.5201 | completed, below baseline and V2.1 |

At this stage, V2.1 is still the strongest completed variant. V4.1 shows that lowering the mask-guidance loss helps, but the mask branch remains below the simpler V2.1 ROI-grid correction. V3.1 shows that simply softening uncertainty weighting does not solve the uncertainty branch; it drops below both V3 and the baseline. V6 tested adaptive query-region fusion and a query-level gate, but it also stayed below V2.1. This means the next stronger direction should focus on region reliability or safer auxiliary supervision, not only a more expressive fusion block.

## Planned Ablations

The original ablation idea was V1-V5. After the first results, the plan became more controlled: first stabilize the ROI-grid depth correction, then add uncertainty or mask guidance only if the simpler branch is useful.

| Name | Description | Current role |
| --- | --- | --- |
| Baseline | Original MonoDGP | reference result |
| V1 | ROI grid 2x2 depth geometry correction | optional size ablation |
| V2 | ROI grid 3x3 depth geometry correction | main branch |
| V2.1 | V2 with `region_geometry_loss_coef=0.05` | current best variant |
| V2.2 | V2 with `region_geometry_loss_coef=0.1` | tuning comparison |
| V3 | ROI grid 3x3 + depth uncertainty weighting | completed, below V2.1 |
| V3.1 | softer uncertainty weighting, no auxiliary uncertainty loss | completed, worse than V3 and baseline |
| V4 | ROI grid 3x3 + region-mask guidance | completed, below baseline |
| V4.1 | V4 with loss coefficient 0.05 | completed, better than V4 but below V2.1 |
| V5 | ROI grid 3x3 + uncertainty + mask | completed, weakest region-aware variant |
| V6 | ROI grid 3x3 + adaptive query-region fusion | completed, below V2.1 |

Fallback variants are also documented in [docs/experiment_matrix.md](docs/experiment_matrix.md). They are useful if direct depth correction becomes unstable again.

## Repository Structure

```text
region-aware-monocular-3d-detection/
|-- configs/
|   |-- monodgp.yaml
|   |-- monodgp_debug.yaml
|   |-- monodgp_baseline_20.yaml
|   `-- variants/
|-- docs/
|   |-- implementation_plan.md
|   |-- experiment_matrix.md
|   |-- experiment_results.md
|   `-- visualization_and_metrics.md
|-- lib/
|   `-- models/
|       `-- monodgp/
|           |-- depth_predictor/
|           |-- region_geometry/
|           |-- region_seg_head.py
|           `-- monodgp.py
|-- tools/
`-- README.md
```

The dissertation contribution code is mainly placed under `lib/models/monodgp/region_geometry/` and connected inside the MonoDGP model.

## Dataset

The project uses the KITTI 3D Object Detection dataset.

Expected local directory layout:

```text
disertation/
|-- region-aware-monocular-3d-detection/
|-- data/
|   `-- kitti/
|       |-- ImageSets/
|       |-- training/
|       |   |-- image_2/
|       |   |-- label_2/
|       |   `-- calib/
|       `-- testing/
|           |-- image_2/
|           `-- calib/
`-- experiments/
    |-- runs/
    |-- metrics/
    `-- visualizations/
```

The default configs use:

```yaml
dataset:
  root_dir: '../data/kitti'
```

## Environment

The current local setup uses:

- Python 3.10
- PyTorch 2.6.0 + CUDA 12.6
- KITTI 3D Object Detection

The custom deformable attention extension must be compiled before training.

## Upstream Citation

This project is based on MonoDGP. If using this code or comparing with MonoDGP, cite the original work:

```bibtex
@inproceedings{pu2025monodgp,
  title={Monodgp: Monocular 3D object detection with decoupled-query and geometry-error priors},
  author={Pu, Fanqi and Wang, Yifan and Deng, Jiru and Yang, Wenming},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={6520--6530},
  year={2025}
}
```

## Acknowledgement

This repository builds on the official MonoDGP implementation and its upstream dependency on MonoDETR.
