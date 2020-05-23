# Experiment all tricks with center loss + new data augmentation
# Dataset: msmt17

python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' DATASETS.NAMES "('msmt17')" OUTPUT_DIR "('./logs')" SOLVER.IMS_PER_BATCH  "(128)" SOLVER.EVAL_PERIOD "(15)" SOLVER.CHECKPOINT_PERIOD "(10)
