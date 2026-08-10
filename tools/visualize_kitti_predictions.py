import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import cv2
import numpy as np

from lib.datasets.kitti.kitti_utils import Calibration, Object3d


BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def read_kitti_objects(path):
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [Object3d(line) for line in lines]


def project_box(obj, calib):
    corners_3d = obj.generate_corners3d()
    if np.any(corners_3d[:, 2] <= 0.1):
        return None
    corners_2d, _ = calib.rect_to_img(corners_3d)
    return corners_2d.astype(int)


def draw_projected_box(image, corners, color, thickness=2):
    for start, end in BOX_EDGES:
        p1 = tuple(corners[start])
        p2 = tuple(corners[end])
        cv2.line(image, p1, p2, color, thickness, lineType=cv2.LINE_AA)


def draw_label(image, text, xy, color):
    x, y = int(xy[0]), int(xy[1])
    y = max(y, 18)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def visualize_sample(image_id, data_root, pred_dir, out_dir, score_threshold, class_name):
    image_path = data_root / 'training' / 'image_2' / f'{image_id:06d}.png'
    calib_path = data_root / 'training' / 'calib' / f'{image_id:06d}.txt'
    gt_path = data_root / 'training' / 'label_2' / f'{image_id:06d}.txt'
    pred_path = pred_dir / f'{image_id:06d}.txt'

    if not image_path.exists() or not calib_path.exists():
        return False

    image = cv2.imread(str(image_path))
    if image is None:
        return False

    calib = Calibration(str(calib_path))
    gt_objects = [obj for obj in read_kitti_objects(gt_path) if obj.cls_type == class_name]
    pred_objects = [obj for obj in read_kitti_objects(pred_path) if obj.cls_type == class_name]
    pred_objects = [obj for obj in pred_objects if obj.score < 0 or obj.score >= score_threshold]

    for obj in gt_objects:
        corners = project_box(obj, calib)
        if corners is not None:
            draw_projected_box(image, corners, (0, 180, 0), thickness=2)
            draw_label(image, 'GT', corners[4], (0, 180, 0))

    for obj in pred_objects:
        corners = project_box(obj, calib)
        if corners is not None:
            draw_projected_box(image, corners, (0, 0, 255), thickness=2)
            label = 'Pred'
            if obj.score >= 0:
                label += f' {obj.score:.2f}'
            draw_label(image, label, corners[0], (0, 0, 255))

    cv2.putText(
        image,
        f'{image_id:06d} | green=GT red=Prediction',
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f'{image_id:06d}.png'), image)
    return True


def parse_image_ids(args, pred_dir):
    if args.image_ids:
        return [int(x) for x in args.image_ids.replace(',', ' ').split()]

    ids = sorted(int(p.stem) for p in pred_dir.glob('*.txt'))
    if args.max_images:
        ids = ids[:args.max_images]
    return ids


def main():
    parser = argparse.ArgumentParser(description='Visualize KITTI 3D projected boxes for MonoDGP predictions.')
    parser.add_argument('--data_root', default='../data/kitti', help='KITTI root folder.')
    parser.add_argument('--pred_dir', required=True, help='Folder containing KITTI prediction txt files, usually outputs/data.')
    parser.add_argument('--out_dir', required=True, help='Folder where rendered images will be saved.')
    parser.add_argument('--image_ids', default='', help='Optional list of image ids, separated by spaces or commas.')
    parser.add_argument('--max_images', type=int, default=20, help='Maximum images when image_ids is not provided.')
    parser.add_argument('--score_threshold', type=float, default=0.2)
    parser.add_argument('--class_name', default='Car')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)

    image_ids = parse_image_ids(args, pred_dir)
    saved = 0
    for image_id in image_ids:
        if visualize_sample(image_id, data_root, pred_dir, out_dir, args.score_threshold, args.class_name):
            saved += 1

    print(f'Saved {saved} visualizations to {out_dir}')


if __name__ == '__main__':
    main()

