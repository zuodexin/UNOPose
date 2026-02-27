import argparse
from glob import glob
import json
from logging import getLogger
import logging
import os

import ipdb
import pandas as pd

logger = getLogger(__name__)

# can, box, bottles, blocks, others
ycbv_objects = [
    {"obj_id": 1, "name": "master_chef_can", "group": "can"},
    {"obj_id": 2, "name": "cracker_box", "group": "box"},
    {"obj_id": 3, "name": "sugar_box", "group": "box"},
    {"obj_id": 4, "name": "tomato_soup_can", "group": "can"},
    {"obj_id": 5, "name": "mustard_bottle", "group": "bottle"},
    {"obj_id": 6, "name": "tuna_fish_can", "group": "can"},
    {"obj_id": 7, "name": "pudding_box", "group": "box"},
    {"obj_id": 8, "name": "gelatin_box", "group": "box"},
    {"obj_id": 9, "name": "potted_meat_can", "group": "can"},
    {"obj_id": 10, "name": "banana", "group": "others"},
    {"obj_id": 11, "name": "pitcher_base", "group": "bottle"},
    {"obj_id": 12, "name": "bleach_cleanser", "group": "bottle"},
    {"obj_id": 13, "name": "bowl", "group": "others"},
    {"obj_id": 14, "name": "mug", "group": "others"},
    {"obj_id": 15, "name": "power_drill", "group": "others"},
    {"obj_id": 16, "name": "wood_block", "group": "block"},
    {"obj_id": 17, "name": "scissors", "group": "others"},
    {"obj_id": 18, "name": "large_marker", "group": "others"},
    {"obj_id": 19, "name": "large_clamp", "group": "others"},
    {"obj_id": 20, "name": "extra_large_clamp", "group": "others"},
    {"obj_id": 21, "name": "foam_brick", "group": "block"},
]

# python -m core.unopose.scripts.group_ycbv_scores --eval_dir output/ycbv_first_cfg/archive/bop_19/inference_model_final/ycbv/resultPfoneref50_ycbv-test
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate standard deviation of a list of numbers."
    )
    parser.add_argument("--eval_dir", type=str, help="Directory of run")
    parser.add_argument("--tag", type=str, help="tag of run", default="cir-ycbv-test")
    args = parser.parse_args()

    eval_dir = args.eval_dir
    # logg to file
    logger.setLevel("INFO")
    handler = logging.FileHandler(f"{args.eval_dir}/group_scores.log")
    handler.setLevel("INFO")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console_handler = logging.StreamHandler()  # 创建控制台处理器
    console_handler.setLevel(logging.INFO)  # 设置处理级别
    console_handler.setFormatter(formatter)  # 使用相同的格式
    logger.addHandler(console_handler)  # 添加到 logger
    metrics = [
        {
            "name": "AUC_ADD",
            "error_sig": "AUCadd",
        },
        {
            "name": "AUC_ADI",
            "error_sig": "AUCadi",
        },
    ]

    scores = []
    for metric in metrics:
        metric_name = metric["name"]
        error_sig = metric["error_sig"]
        score_paths = glob(
            f"{eval_dir}/error={error_sig}*/scores*.json"
        )
        for score_path in score_paths:
            with open(score_path, "r") as f:
                data = json.load(f)
                obj_recalls = data["obj_recalls"]
                for obj_id, score in obj_recalls.items():
                    scores.append(
                        dict(
                            metric=metric_name,
                            obj_id=int(obj_id),
                            score=score,
                        )
                    )

    scores_df = pd.DataFrame(scores)
    grouped_scores = scores_df.groupby(["metric", "obj_id"]).mean().reset_index()

    grouped_scores["group"] = grouped_scores["obj_id"].apply(
        lambda x: next(
            (obj["group"] for obj in ycbv_objects if obj["obj_id"] == x), "unknown"
        )
    )

    grouped_scores = grouped_scores.groupby(["metric", "group"]).mean().reset_index()
    # sort by group ["can", "box", "bottle", "block", "others"]
    group_order = ["can", "box", "bottle", "block", "others"]
    grouped_scores["group"] = pd.Categorical(
        grouped_scores["group"], categories=group_order, ordered=True
    )
    grouped_scores = grouped_scores.sort_values(by=["metric", "group"]).reset_index(
        drop=True
    )

    logger.info(f"Grouped scores:\n{grouped_scores}")

    # remove obj_id column
    grouped_scores = grouped_scores.drop(columns=["obj_id"])
    # mean scores
    mean_scores = (
        grouped_scores.drop(columns=["group"]).groupby("metric").mean().reset_index()
    )
    logger.info(f"Mean scores:\n{mean_scores}")

    print("logs saved to", f"{eval_dir}/group_scores.log")
