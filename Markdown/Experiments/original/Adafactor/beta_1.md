# Beta_1

## Comments

-   beta1 makes no difference

## Dataset

-   Wikitext-2

## Model types

-   gpt2-medium - 355M parameters

## Hyperparameters

| seq len | epochs | global batch size | optimizer | learning rate |
| ------- | ------ | ----------------- | --------- | ------------- |
| 256     | 10     | 64                | Adafactor | 5e-5          |

## Adafactor

-   beta_1: 0.0
-   beta_1: 0.9

## Results

_Selected on best val score_

| beta_1 | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run                     |
| ------ | ---------- | -------- | ------- | ----------- | ---------- | --------- | ----------------------- |
| 0      | -          | 2.954    | 19.413  | 29.769      | 3          | TF        | laced-mountain-712      |
| 0.9    | -          | 2.955    | 19.429  | 29.88       | 3          | TF        | celestial-firebrand-734 |
