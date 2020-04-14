# Wikitext 103

## Comments

-   wikitext 103 requires a lot larger batch sizes than smaller datasets like wikitext2
-   still overfits after 1-2 epochs on distilgpt2
-   batch size 128 and 64 too small

## Dataset

-   Wikitext-103

## Model types

-   gpt2-medium - largest batch size 256
-   gpt2-large - largest batch size 128
-   gpt2-xl - largest batch size 64

## Hyperparameters

| batch size | seq len | epochs | optimizer | learning rate |
| ---------- | ------- | ------ | --------- | ------------- |
| 128        | 256     | 10     | Adafactor | 5e-5          |

## Adafactor

-   beta_1: 0.0

## Results

_Selected on best val score_

| model                   | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run               |
| ----------------------- | ---------- | -------- | ------- | ----------- | ---------- | --------- | ----------------- |
| gpt2-medium             | -          | 2.945    | 19.169  | 29.327      | 0          | TF        | colorful-surf-776 |
| gpt2-large              | -          | 3.033    | 21.014  | 32.596      | 0          | TF        | stilted-wood-777  |
| gpt2-xl (batch size 64) | -          | 3.452    | 32.487  | 53.792      | 0          | TF        | giddy-oath-778    |
