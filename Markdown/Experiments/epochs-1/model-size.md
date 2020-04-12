# 1 epoch

## Comments

-   you can train on small datasets in 1 epoch
-   reduce batch size to counteract to small number of iterations

## Dataset

-   Wikitext-2

## Model types

-   distilgpt2 - 82M parameters
-   gpt2 - 124M parameters
-   gpt2-medium - 355M parameters
-   gpt2-large - 774M parameters
-   gpt2-xl - 1.5B parameters

## Hyperparameters

| seq len | epochs | global batch size | optimizer | learning rate |
| ------- | ------ | ----------------- | --------- | ------------- |
| 256     | 1      | 8                 | Adafactor | 5e-5          |

## Adafactor

-   beta_1: 0.0

## Results

_Selected on best val score_

| model       | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run                 |
| ----------- | ---------- | -------- | ------- | ----------- | ---------- | --------- | ------------------- |
| distilgpt2  |
| gpt2        |
| gpt2-medium | -          | 2.971    | 20.875  | 32.641      | 0          | TF        | clear-puddle-793    |
| gpt2-large  | -          | 2.902    | 19.56   | 30.321      | 0          | TF        | cerulean-sunset-794 |
| gpt2-xl     | -          | 2.838    | 18.419  | 28.323      | 0          | TF        | morning-vortex-795  |
