#!/bin/bash
k=41
# datasets=("OITAVEN" "XESTA" "EIRAS" "ERMIDAS" "FERREIRAS" "MESTAS" "MERA" "ULLA")
datasets=("ULLA")


for((i=0; i<${#datasets[@]}; i++)) do
    dataset=${datasets[i]}
    echo "------ Experimenting with $dataset ------"
    bash 3_each_dataset.sh $dataset $k
    k=$((k+1))
done