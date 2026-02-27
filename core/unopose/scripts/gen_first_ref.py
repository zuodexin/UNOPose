import os
import argparse

import cv2
import ipdb
from matplotlib import pyplot as plt
import skimage
from tqdm import tqdm
import os.path as osp
from bop_toolkit_lib import inout


from third_party.bop_toolkit.bop_toolkit_lib.dataset_params import (
    get_model_params,
    get_present_scene_ids,
    get_split_params,
)

# python -m core.unopose.scripts.gen_first_ref
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bop_root", type=str, default="datasets/BOP_DATASETS")
    parser.add_argument("--dataset", type=str, default="lm")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--split_type", type=str, default="none")
    parser.add_argument("--bop19_test", action="store_true")

    args = parser.parse_args()
    bop_root = args.bop_root
    dataset = args.dataset
    split = args.split
    split_type = args.split_type if args.split_type != "none" else None

    dp_split = get_split_params(bop_root, dataset, split, split_type)

    dp_eval_model = get_model_params(bop_root, dataset, "eval")

    scene_ids = get_present_scene_ids(dp_split)
    obj_ids = dp_eval_model["obj_ids"]
    
    if args.bop19_test:
        test_target = inout.load_json(osp.join(dp_split["base_path"], "test_targets_bop19.json"))
        valid_target = {}
        for item in test_target:
            valid_target.setdefault(item["scene_id"],{}).setdefault(item["im_id"], {}).setdefault(item["obj_id"], 1)
    ref_targets = []

    first_view = {}
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
                obj_id = inst_gt["obj_id"]
                if (obj_id, scene_id) not in first_view:
                    first_view[(obj_id, scene_id)] = im_id
                ref_target = dict(
                    scene_id=scene_id,
                    im_id=im_id,
                    obj_id=obj_id,
                    inst_count=1,
                    ref_im_id=first_view[(obj_id, scene_id)],
                    ref_scene_id=scene_id,
                )
                if args.bop19_test:
                    if not valid_target.get(scene_id, {}).get(im_id, {}).get(obj_id, {}):
                        continue
                ref_targets.append(ref_target)
    save_path = os.path.join(
        args.bop_root,
        dataset,
        "test_ref_targets_firstview.json",
    )
    inout.save_json(save_path, ref_targets)
    print(f"Saved first view ref targets to {save_path}")
