export PYTHONPATH=$PYTHONPATH:./third_party/bop_toolkit

GPU_IDs=0
CKPT=checkpoints/model_final.pth
CFG=ycbv_first_cfg
DATASET=ycbv
PROJ_DIR="."

./core/unopose/save_unopose.sh configs/${CFG}.py ${GPU_IDs} ${CKPT}


python ${PROJ_DIR}/core/unopose/engine/bop_eval_utils.py \
--script-path third_party/bop_toolkit/scripts/eval_pose_results_more.py \
--targets_name test_targets_keyframes.json \
--error_types 'vsd,mssd,mspd,ad,rete,AUCadd,AUCadi,AUCad' \
--split test \
--dataset ${DATASET} \
--result_names resultPfoneref50_${DATASET}-test.csv \
--result_dir output/${CFG}/inference_model_final/${DATASET}
