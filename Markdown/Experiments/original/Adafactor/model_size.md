# Model size

https://app.wandb.ai/bkkaggle/lm-finetuning/reports/model-size-with-adafactor--Vmlldzo3MTA4MQ

## Comments

Adafactor performs slightly worse that AdamW

On pytorch, Adafactor takes more epochs (7-10 compared to 1-2; on wikitext-2) than AdamW (possibly due to beta1 bug)

Diminishing returns after gpt2-medium

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
| 256     | 10     | 16                | Adafactor | 5e-5          |

## Adafactor

-   beta_1: 0.0

## Results

_Selected on best val score_

| model       | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run                  |
| ----------- | ---------- | -------- | ------- | ----------- | ---------- | --------- | -------------------- |
| distilgpt2  | -          | 3.522    | 35.234  | 59.133      | 0          | TF        | neat-wood-673        |
| gpt2        | -          | 3.28     | 27.639  | 44.799      | 1          | TF        | wobbly-dawn-695      |
| gpt2-medium | -          | 2.994    | 20.637  | 32.041      | 0          | TF        | morning-feather-696  |
| gpt2-large  | -          | 2.887    | 18.631  | 28.521      | 0          | TF        | proud-firebrand-697  |
| gpt2-xl     | -          | 2.811    | 17.255  | 26.121      | 0          | TF        | devout-butterfly-703 |
