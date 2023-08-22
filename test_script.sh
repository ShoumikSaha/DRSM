#!/bin/sh
#SBATCH --ntasks=1 --cpus-per-task=4 --mem-per-cpu=16G --time=24:00:00

export PYTHONPATH="$PYTHONPATH:$PWD"

python train_custom_malconv.py --mal_path '../Dataset/malware/' --ben_path '../Dataset/benign' --save_path 'secml_malware/data/trained/custom_malconv_1_2mb.dat' --dataset_size 2000 --ablations 1 --epochs 2 --batch_size 8
