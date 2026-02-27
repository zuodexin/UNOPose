import argparse
from glob import glob
import json
from logging import getLogger
import logging
import ipdb
import pandas as pd

logger = getLogger(__name__)

ycbv_obj_id2name = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}
ycbv_name2obj_id = {v: k for k, v in ycbv_obj_id2name.items()}

# python -m core.unopose.scripts.report_ycbv_auc_add --eval_dir output/ycbv_first_cfg/inference_model_final/ycbv/resultPfoneref50_ycbv-test
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YCBV Object ID to Name Mapping")
    parser.add_argument("--eval_dir", type=str, help="Directory of run")
    parser.add_argument("--tag", type=str, help="tag of run", default="cir-ycbv-test")
    args = parser.parse_args()

    eval_dir = args.eval_dir
    # logg to file
    logger.setLevel("INFO")
    handler = logging.FileHandler(f"{args.eval_dir}/scores_obj_id.log")
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

    grouped_scores["obj_name"] = grouped_scores["obj_id"].apply(
        lambda x: ycbv_obj_id2name[x]
    )
    grouped_scores = grouped_scores.sort_values(by=["metric", "obj_id"]).reset_index(
        drop=True
    )
    logger.info(f"scores:\n{grouped_scores}")
    logger.info(scores_df.groupby("metric").mean())
    logger.info(f"Scores saved to {eval_dir}/scores_obj_id.log")
