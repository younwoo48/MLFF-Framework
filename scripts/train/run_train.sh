#!/bin/bash

GPU=0

MODEL_ARRAY=(
    "BPNN"
    "SchNet"
    "DimeNet++"
    "GemNet-T"
    "GemNet-dT"
    "NequIP"
    "Allegro"
    "MACE"
    "SCN"
)
MODEL=GemNet-T

DATA_ARRAY=(
    "SiN"
    "HfO"
)
DATA=HfO

BENCHMARK_HOME=$(realpath ../../)
cd $BENCHMARK_HOME

CONFIG=configs/train/${DATA}/${MODEL}.yml
RUNDIR=train_results/${DATA}/${MODEL}
RUNID=train

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $RUNDIR \
    --identifier $RUNID \
    --print-every 100 \
    --save-ckpt-every-epoch 20 \


