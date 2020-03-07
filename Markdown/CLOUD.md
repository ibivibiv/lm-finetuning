# Cloud

## Setup

```bash
git config --global credential.helper cache
git config --global credential.helper 'cache --timeout=84000'
git config --global user.email "bk@tinymanager.com"
git config --global user.name "Bilal Khan"
git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir ./ && cd .. && rm -rf apex
git clone https://bkkaggle:5e3fc022d0e763c083f23918224f2bc17e95ec2e@github.com/bkkaggle/lm-finetuning.git
```

## Usage

```bash
export XRT_TPU_CONFIG="tpu_worker;0;10.95.13.146:8470"
export COLAB_GPU=1
conda activate torch-xla-nightly
. /usr/share/torch-xla-nightly/pytorch/xla/scripts/update_nightly_torch_wheels.sh
```
