#!/usr/bin/env bash

for i in `seq 19 10 399`; do

    ckpt_file="checkpoint/ComplEx-$i.ckpt"
    echo "Testing checkpoint $ckpt_file"

    cd result
    rm ComplEx.ckpt
    ln -s ../$ckpt_file ComplEx.ckpt
    cd ..
    python train_and_export_fb15k-237z_complex300.py > ../pretrained_models/fb15k-237-owe/complex300/results_${i}.txt
    tail -n 19 ../pretrained_models/fb15k-237-owe/complex300/results_${i}.txt
done