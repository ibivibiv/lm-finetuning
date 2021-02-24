# Language Modeling Finetuning

## set up 

run 

pip install git+https://github.com/bkkaggle/transformers.git@albert-style
pip install wandb

## run sample
python3 train_pt.py --sample_only --tokenizer ./algpt2/algpt2-tokenizer/ --seq_len 1024 --checkpoint ./checkpoint-batch-78999/tf_model.h5 --config ./final_epoch/config.json --from_tf --do_sample --temperature 1 --top_p 0.8 --n_samples 

## Tools

-   paperswithcode
-   https://www.groundai.com
-   google scholar
-   arxiv-sanity
-   research gate
-   openreview

## Acknowledgements

Some code is taken from https://github.com/uber-research/PPLM, https://github.com/salesforce/ctrl, https://github.com/huggingface/transformers, and https://github.com/bkkaggle/lm-experiments
