#!/bin/bash

DATA=$1

BENCHMARK_HOME=$(realpath ../../)
DATADIR=${BENCHMARK_HOME}/datasets/${DATA}
OUTDIR=${BENCHMARK_HOME}/datasets/${DATA}

# If you want to prepare .lmdb which saves just atom cloud (containing just coordinates), set "cloud".
# Or if you want to have graph (containing coordinates as well as edges), set "graph"
outdata_type=$2

if [ $outdata_type == "cloud" ]; then

# Train/Valid/Test sets
python preprocess.py \
    --train-data ${DATADIR}/Trainset.xyz \
    --train-data-output-name train \
    --valid-data ${DATADIR}/Validset.xyz \
    --valid-data-output-name valid \
    --test-data ${DATADIR}/Testset.xyz \
    --test-data-output-name test \
    --out-path ${OUTDIR} \
    --energy-type free_energy \

# OOD
python preprocess.py \
    --data ${DATADIR}/OOD.xyz \
    --data-output-name ood \
    --out-path ${OUTDIR}/ood \

elif [ $outdata_type == "graph" ]; then

rmax=$3
maxneigh=$4

# Train/Valid/Test sets
python preprocess.py \
    --train-data ${DATADIR}/Trainset.xyz \
    --train-data-output-name train \
    --valid-data ${DATADIR}/Validset.xyz \
    --valid-data-output-name valid \
    --test-data ${DATADIR}/Testset.xyz \
    --test-data-output-name test \
    --out-path ${OUTDIR} \
    --r-max $rmax \
    --max-neighbors $maxneigh \

# OOD
python preprocess.py \
    --data ${DATADIR}/OOD.xyz \
    --data-output-name ood \
    --out-path ${OUTDIR}/ood \
    --r-max $rmax \
    --max-neighbors $maxneigh \

fi
