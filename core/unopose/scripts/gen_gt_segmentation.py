import os
import argparse

import cv2
import ipdb
from matplotlib import pyplot as plt
import numpy as np
import skimage
from tqdm import tqdm
from bop_toolkit_lib import inout


from lib.utils.mask_utils import binary_mask_to_rle
from third_party.bop_toolkit.bop_toolkit_lib.dataset_params import (
    get_model_params,
    get_present_scene_ids,
    get_split_params,
)

# python -m core.unopose.scripts.gen_gt_segmentation
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bop_root", type=str, default="datasets/BOP_DATASETS")
    parser.add_argument("--dataset", type=str, default="lm")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--split_type", type=str, default="none")

    args = parser.parse_args()
    bop_root = args.bop_root
    dataset = args.dataset
    split = args.split
    split_type = args.split_type if args.split_type != "none" else None

    dp_split = get_split_params(bop_root, dataset, split, split_type)

    dp_eval_model = get_model_params(bop_root, dataset, "eval")

    scene_ids = get_present_scene_ids(dp_split)
    obj_ids = dp_eval_model["obj_ids"]

    segmentations = []

    for scene_id in tqdm(scene_ids):
        gt_info = inout.load_scene_gt_info(
            dp_split["scene_gt_info_tpath"].format(scene_id=scene_id)
        )
        scene_gt = inout.load_scene_gt(
            dp_split["scene_gt_tpath"].format(scene_id=scene_id)
        )
        for im_id in sorted(scene_gt.keys()):
            info = gt_info[im_id]
            for gt_id, inst_gt in enumerate(scene_gt[im_id]):
                mask = (
                    cv2.imread(
                        dp_split["mask_visib_tpath"].format(
                            scene_id=scene_id, im_id=im_id, gt_id=gt_id
                        )
                    )[:, :, 0]
                    > 0
                )
                mask_rle = binary_mask_to_rle(mask, compressed=False)
                bbox = info[gt_id]["bbox_visib"].astype(np.int32).tolist()

                segmentation = dict(
                    scene_id=scene_id,
                    image_id=im_id,
                    category_id=inst_gt["obj_id"],
                    bbox=bbox,
                    score=1.0,
                    time=1.0,
                    segmentation=mask_rle,
                )
                segmentations.append(segmentation)
    out_path = os.path.join(
        "datasets/segmentation",
        f"gt_segmentation_{dataset}_{split}{'_' + split_type if split_type is not None else ''}.json",
    )
    inout.save_json(out_path, segmentations)
