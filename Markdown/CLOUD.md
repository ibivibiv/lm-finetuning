# Cloud

## Setup

```bash
git config --global credential.helper cache
git config --global credential.helper 'cache --timeout=84000'
git config --global user.email "bk@tinymanager.com"
git config --global user.name "Bilal Khan"
pip install wandb git+https://github.com/bkkaggle/transformers.git git+https://github.com/PyTorchLightning/pytorch-lightning.git@master
python -m pip install --upgrade ptvsd
git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir ./ && cd .. && rm -rf apex
git clone https://bkkaggle:5e3fc022d0e763c083f23918224f2bc17e95ec2e@github.com/bkkaggle/lm-finetuning.git
wandb login 0133b27327cda5d706c51225880c900e9b6878fb
```

### TPU

```bash
git config --global credential.helper cache
git config --global credential.helper 'cache --timeout=84000'
git config --global user.email "bk@tinymanager.com"
git config --global user.name "Bilal Khan"
pip3 install wandb
pip3 install --user git+https://github.com/bkkaggle/transformers.git tensorflow_datasets torch
git clone https://bkkaggle:5e3fc022d0e763c083f23918224f2bc17e95ec2e@github.com/bkkaggle/lm-finetuning.git
wandb login 0133b27327cda5d706c51225880c900e9b6878fb
```

## Usage

```bash
export XRT_TPU_CONFIG="tpu_worker;0;10.95.13.146:8470"
export COLAB_GPU=1
conda activate torch-xla-nightly
. /usr/share/torch-xla-nightly/pytorch/xla/scripts/update_nightly_torch_wheels.sh
```

### TF

```bash
export COLAB_TPU_ADDR="10.182.32.10:8470"
```

SSH in directly with `ssh -i ~/.ssh/google_compute_engine 34.91.232.203`
