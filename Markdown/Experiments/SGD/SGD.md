# SGD

## Comments

-   SGD varies a lot over multiple runs
-   using an lr-scheduler lets you get only a little higher (~1ppl higher) than adafactor
-   SGD uses slightly less memory and takes 10x longer

## Dataset

-   Wikitext-2

## Learning rates

-   1e-2
-   warmup + decay schedule

## Hyperparameters

| model       | batch size | epochs | optimizer |
| ----------- | ---------- | ------ | --------- |
| gpt2-medium | 64         | 10     | SGD       |

## Results

_Selected on best val score_

| learning rate | warmup + decay | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run                  |
| ------------- | -------------- | ---------- | -------- | ------- | ----------- | ---------- | --------- | -------------------- |
| 1e-2          | No             | -          | 4.758    | 117.486 | 233.161     | 9          | TF        | fearless-feather-729 |
| 1e-2          | Yes            | -          | 3.071    | 21.765  | 33.194      | 9          | TF        | polished-jazz-732    |
