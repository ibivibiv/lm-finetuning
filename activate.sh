#!/bin/bash

export XRT_TPU_CONFIG="tpu_worker;0;10.95.13.146:8470"
export COLAB_GPU=1

conda activate torch-xla-nightly

cd /usr/share/torch-xla-nightly/pytorch/xla && . ./scripts/update_nightly_torch_wheels.sh