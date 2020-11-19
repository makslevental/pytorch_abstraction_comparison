#!/bin/bash
# first arg is executable, second arg is cuda device
set -e
for dataset in "pascal" "mnist" "cifar10" "stl10"; do
  for run in 0 1 2 3 4 5 6 7 8 9; do
    CUDA_VISIBLE_DEVICES=$2 ./build/bin/main_$1 $dataset $run
  done
done
