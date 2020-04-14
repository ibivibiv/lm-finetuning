# Batch Size

https://app.wandb.ai/bkkaggle/lm-finetuning/reports/batch-size-with-adafactor-seqlen-256--Vmlldzo3MTA4Ng

## Comments

-   check relation between batch size and dataset size

-   larger batch size
    -   trains faster
    -   trains to a lower loss

## Dataset

-   Wikitext-2

## Batch Sizes

-   16
-   32
-   64

## Hyperparameters

| model       | seq len | epochs | optimizer | learning rate |
| ----------- | ------- | ------ | --------- | ------------- |
| gpt2        | 256     | 10     | Adafactor | 5e-5          |
| gpt2-medium | 256     | 10     | Adafactor | 5e-5          |

## Adafactor

-   beta_1: 0.0

## Results

_Selected on best val score_

| batch size | model       | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run                 |
| ---------- | ----------- | ---------- | -------- | ------- | ----------- | ---------- | --------- | ------------------- |
| 16         | gpt2        | -          | 3.28     | 27.639  | 44.799      | 1          | TF        | wobbly-dawn-695     |
| 32         | gpt2        | -          | 3.242    | 26.147  | 41.918      | 2          | TF        | unique-lake-708     |
| 64         | gpt2        | -          | 3.22     | 25.303  | 40.3        | 7          | TF        | decent-salad-710    |
| 16         | gpt2-medium | -          | 2.994    | 20.637  | 32.041      | 0          | TF        | morning-feather-696 |
| 32         | gpt2-medium | -          | 2.966    | 19.837  | 30.561      | 1          | TF        | faithful-puddle-711 |
| 64         | gpt2-medium | -          | 2.954    | 19.413  | 29.769      | 3          | TF        | laced-mountain-712  |
