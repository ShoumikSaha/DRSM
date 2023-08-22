#!/bin/sh

ROOT_DIR="$1"
TRAIN_CSV="$2"
VAL_CSV="$3"
TEST_CSV="$4"
BATCH_SIZE="$5"

for (( i=0; i<16; i++ ))
do
	echo "$i"
	python train_custom_malconv_by_ablation_from_csv.py --root_dir "$ROOT_DIR" --train_path "$TRAIN_CSV" --val_path "$VAL_CSV" --test_path "$TEST_CSV" --dir_path 'secml_malware/data/trained/smoothed_8' --ablation_idx $i --ablations 16 --epochs 5 --batch_size $BATCH_SIZE >> output/train_16_output.txt
done
