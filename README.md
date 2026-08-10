# Region-Aware Monocular 3D Detection

This repository contains the implementation work for a dissertation project in monocular 3D object detection. It is based on [MonoDGP](https://github.com/PuFanqi23/MonoDGP), which is used as the baseline model.

The dissertation direction is:

> Region-aware uncertainty-guided geometry-error estimation for monocular 3D object detection.

## Motivation

Monocular 3D object detection is difficult because the model has to infer 3D geometry from a single RGB image. Depth, object location, size, and orientation are all ambiguous, especially for occluded, truncated, or distant objects.

MonoDGP addresses this problem using decoupled queries and geometry-error priors. This project keeps MonoDGP as the baseline and studies whether geometry error can be modeled more locally, at region level, instead of only at global object/query level.

The main hypothesis is that different regions of the same object may have different geometric reliability. Some regions may provide useful cues for the final 3D box, while others may be affected by occlusion, truncation, background noise, or uncertain depth.

## Proposed Direction

The proposed extension introduces a region-aware geometry-error estimation framework. For each predicted object query, the predicted 2D box is used as an ROI on the feature map. A small 2x2 or 3x3 grid is then pooled from inside that object area, and each local cell can contribute to a depth correction.

These local signals can then be aggregated before the final 3D box prediction. Depth uncertainty and region masks may also be used to weight the contribution of each region.

## Ablation Variants

The first implementation stage is an ablation study with five variants. These variants are tested before choosing the next dissertation direction.

| Name | Description |
| --- | --- |
| Baseline | Original MonoDGP. |
| V1 | ROI grid 2x2 region-aware depth geometry correction. |
| V2 | ROI grid 3x3 region-aware depth geometry correction. |
| V3 | ROI grid 3x3 + depth uncertainty weighting. |
| V4 | ROI grid 3x3 + region-mask guidance. |
| V5 | ROI grid 3x3 + depth uncertainty weighting + region-mask guidance. |

The final model is not fixed in advance. V1-V5 should be compared first, then the next step should be chosen based on validation results, stability, and complexity.

The current V1-V5 configs use a learnable depth gate. The gate starts from zero, so the model initially behaves like the MonoDGP baseline and only learns to use the regional depth correction if it is useful.
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
|           |-- region_seg_head.py
|           |-- depth_predictor/
|           |-- region_geometry/
|           `-- monodgp.py
|-- tools/
`-- README.md
```

`region_geometry/` is reserved for the dissertation contribution. The baseline code remains in the original MonoDGP files until a variant is explicitly connected.

Experiment results should be tracked in `docs/experiment_results.md` after every run.

## Main Changes From Upstream MonoDGP

The current repository includes compatibility updates needed to run MonoDGP on a newer local environment:

- PyTorch 2.x compatibility updates for the custom CUDA attention extension.
- CUDA architecture update for newer NVIDIA GPUs.
- Updated imports for PyTorch internal API changes.
- Numba compatibility fixes for KITTI evaluation.
- Checkpoint loading update for newer `torch.load` behavior.
- Debug and preliminary baseline configs.

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
    `-- runs/
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




