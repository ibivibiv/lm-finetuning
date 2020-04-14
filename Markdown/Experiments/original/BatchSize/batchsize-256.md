# Batch Size 256

## Comments

-   larger batch size is almost always better, but improves for more epochs than at a smaller batch size
-   So far, this has only been tested for wikitext2

## Dataset

-   Wikitext-2

## Models

-   distilgpt2
-   gpt2
-   gpt2-medium

## Hyperparameters

| batch size | seq len | epochs | optimizer | learning rate |
| ---------- | ------- | ------ | --------- | ------------- |
| 256        | 256     | 10     | Adafactor | 5e-5          |

## Adafactor

-   beta_1: 0.0

## Results

_Selected on best val score_

| model       | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run                      |
| ----------- | ---------- | -------- | ------- | ----------- | ---------- | --------- | ------------------------ |
| distilgpt2  | -          | 3.458    | 31.892  | 52.45       | 9          | TF        | smart-donkey-788         |
| gpt2        | -          | 3.205    | 24.752  | 39.257      | 5          | TF        | sparkling-wildflower-789 |
| gpt2-medium | -          | 2.938    | 18.635  | 28.463      | 5          | TF        | playful-breeze-790       |
