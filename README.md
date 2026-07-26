# Region-Aware Monocular 3D Detection

This repository contains the implementation work for a dissertation project in
monocular 3D object detection. It is based on
[MonoDGP](https://github.com/PuFanqi23/MonoDGP), which is used as the baseline
model.

The dissertation direction is:

> Region-aware uncertainty-guided geometry-error estimation for monocular 3D
> object detection.

## Motivation

Monocular 3D object detection is difficult because the model has to infer 3D
geometry from a single RGB image. Depth, object location, size, and orientation
are all ambiguous, especially for occluded, truncated, or distant objects.

MonoDGP addresses this problem using decoupled queries and geometry-error
priors. This project keeps MonoDGP as the baseline and studies whether geometry
error can be modeled more locally, at region level, instead of only at global
object/query level.

The main hypothesis is that different regions of the same object may have
different geometric reliability. Some regions may provide useful cues for the
final 3D box, while others may be affected by occlusion, truncation, background
noise, or uncertain depth.

## Proposed Direction

The proposed extension introduces a region-aware geometry-error estimation
framework. The object area is divided into local regions, and each region can
produce a geometry-error score, confidence score, or small geometric correction.

These local signals can then be aggregated before the final 3D box prediction.
Depth uncertainty and region masks may also be used to weight the contribution
of each region.

## Planned Variants

The project is organized around a MonoDGP baseline and several region-aware
variants.

| Name | Description |
| --- | --- |
| Baseline | Original MonoDGP. |
| Variant A | MonoDGP + region-aware geometry-error head. |
| Variant B | Variant A + uncertainty-guided region weighting. |
| Variant C | Region-aware geometry error + uncertainty + auxiliary region loss. |
| Ablation | Compare different region definitions and remove individual components. |

The technical candidate variants are:

| Candidate | Region definition |
| --- | --- |
| V1 | Grid 2x2 region geometry-error map. |
| V2 | Grid 3x3 region geometry-error map. |
| V3 | Grid 3x3 + depth uncertainty weighting. |
| V4 | Grid 3x3 + region mask guidance. |
| V5 | Grid 3x3 + depth uncertainty + region mask. |

The final model is not fixed in advance. It will be selected based on validation
results, stability, and architectural complexity.

## Main Changes From Upstream MonoDGP

The current repository includes compatibility updates needed to run MonoDGP on a
newer local environment:

- PyTorch 2.x compatibility updates for the custom CUDA attention extension.
- CUDA architecture update for newer NVIDIA GPUs.
- Updated imports for PyTorch internal API changes.
- Numba compatibility fixes for KITTI evaluation.
- Checkpoint loading update for newer `torch.load` behavior.
- Debug and preliminary baseline configs.

## Dataset

The project uses the KITTI 3D Object Detection dataset.

Expected directory layout:

```text
disertation/
├── MonoDGP/
└── data/
    └── kitti/
        ├── ImageSets/
        ├── training/
        │   ├── image_2/
        │   ├── label_2/
        │   └── calib/
        └── testing/
            ├── image_2/
            └── calib/
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

This project is based on MonoDGP. If using this code or comparing with MonoDGP,
cite the original work:

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

This repository builds on the official MonoDGP implementation and its upstream
dependency on MonoDETR.
