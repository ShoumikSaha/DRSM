#!/bin/sh

ROOT_DIR="$1"
TRAIN_CSV="$2"
VAL_CSV="$3"
TEST_CSV="$4"

python evaluate_custom_malconv_cert_acc.py --root_dir "$ROOT_DIR" --train_path "$TRAIN_CSV" --val_path "$VAL_CSV" --test_path "$TEST_CSV" --dir_path 'secml_malware/data/trained/smoothed_4' --ablations 4 --batch_size 64 --perturb_size 10000,20000,50000,100000,150000,200000,250000,300000,350000,400000,450000,500000 >> output/cert_4_output.txt
