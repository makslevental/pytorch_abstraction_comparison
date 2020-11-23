#!/bin/bash
# first arg is executable, second arg is cuda device
set -e
for dataset in "pascal"; do
  for run in 0; do
    for res in 32 64 128 256 512 1024 2048 4096; do
        echo BATCH_SIZE=1 EPOCHS=1 RESOLUTION=$res CUDA_VISIBLE_DEVICES=$2 ./build/bin/main_$1 $dataset $run $res
        BATCH_SIZE=1 EPOCHS=1 RESOLUTION=$res CUDA_VISIBLE_DEVICES=$2 ./build/bin/main_$1 $dataset $run $res
        #CUDA_VISIBLE_DEVICES=$2 ./build/bin/main_$1 $dataset $run
    done 
  done
done
