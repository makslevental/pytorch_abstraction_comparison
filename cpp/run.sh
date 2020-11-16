#!/bin/bash
# first arg is executable, second arg is which run, third arg is cuda device
set -e
for i in 0 1 2 3 4 5 6 7 8 9; do
  CUDA_VISIBLE_DEVICES=$3 nvprof -o profiles/prof_${1}_${2}_${i}.nvvp ./build/bin/main_$1 $2 $i
done
