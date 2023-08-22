#!/bin/sh

ROOT_DIR="$1"
TRAIN_CSV="$2"
VAL_CSV="$3"
TEST_CSV="$4"
EPOCHS="$5"

for (( i=0; i<1; i++ ))
do
	echo "$i"
	python train_custom_malconv_by_ablation_from_csv.py --root_dir "$ROOT_DIR" --train_path "$TRAIN_CSV" --val_path "$VAL_CSV" --test_path "$TEST_CSV" --dir_path 'secml_malware/data/trained/smoothed_1' --ablation_idx $i --ablations 1 --epochs $EPOCHS --batch_size 16 >> output/train_1_output.txt
done
