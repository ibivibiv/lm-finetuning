# IMDB

https://app.wandb.ai/bkkaggle/lm-finetuning/reports/imdb--Vmlldzo3NDUyMA/edit

## Comments

-   Training on IMDB is harder
-   not as much improvements for larger models

## Dataset

-   IMDB
-   90-10 train-val split

## Model types

-   distilgpt2
-   gpt2
-   gpt2-medium
-   gpt2-large
-   gpt2-xl

## Hyperparameters

| batch size | seq len | epochs | optimizer | learning rate |
| ---------- | ------- | ------ | --------- | ------------- |
| 64         | 256     | 1      | Adafactor | 5e-5          |

## Adafactor

-   beta_1: 0.0

## Results

_Selected on best val score_

| model       | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run                |
| ----------- | ---------- | -------- | ------- | ----------- | ---------- | --------- | ------------------ |
| gpt2-medium | -          | 3.334    | 28.25   | 73.426      | 0          | TF        | firm-dream-798     |
| gpt2-large  | -          | 3.246    | 25.878  | 65.604      | 0          | TF        | unique-plasma-799  |
| gpt2-xl     | -          | 3.183    | 24.267  | 60.396      | 0          | TF        | dauntless-tree-800 |
