export PYTHONPATH=$PYTHONPATH:./third_party/bop_toolkit
python ./core/unopose/engine/bop_eval_utils.py --script-path third_party/bop_toolkit/scripts/eval_pose_results_more.py \
--targets_name test_targets_bop19.json \
--error_types 'vsd,mssd,mspd,ad,rete' \
--split test \
--dataset tless \
--result_names resultPfoneref50_tless-test_primesense.csv \
--result_dir output/tless_cfg/inference_model_final/tless