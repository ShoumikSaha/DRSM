#!/bin/sh

ROOT_DIR="$1"
TRAIN_CSV="$2"
VAL_CSV="$3"
TEST_CSV="$4"


python evaluate_custom_malconv_by_ablation_from_csv.py --root_dir "$ROOT_DIR" --train_path "$TRAIN_CSV" --val_path "$VAL_CSV" --test_path "$TEST_CSV" --dir_path 'secml_malware/data/trained/smoothed_4' --ablations 4 --batch_size 16 --perturb_size 20000 >> output/eval_4_output.txt
