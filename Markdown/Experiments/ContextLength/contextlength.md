# Context length

## Comments

-   can't train gpt2-xl with a batch size of 16 on 8 cores, need to use a tpuv3-32 with a batch size of 32
    -   using batch-size 8 for gpt2-xl for now

## Dataset

-   Wikitext-2

## Context Len

-   256
-   512
-   1024

## Hyperparameters

| model       | batch size | epochs | optimizer | learning rate |
| ----------- | ---------- | ------ | --------- | ------------- |
| gpt2        | 16         | 10     | Adafactor | 5e-5          |
| gpt2-medium | 16         | 10     | Adafactor | 5e-5          |
| gpt2-xl     | 8          | 10     | Adafactor | 5e-5          |

## Results

_Selected on best val score_

| context length | model       | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run                  |
| -------------- | ----------- | ---------- | -------- | ------- | ----------- | ---------- | --------- | -------------------- |
| 256            | gpt2        | -          | 3.28     | 27.639  | 44.799      | 1          | TF        | wobbly-dawn-695      |
| 512            | gpt2        | -          | 3.13     | 23.525  | 37.446      | 2          | TF        | valiant-glitter-714  |
| 1024           | gpt2        | -          | 3.018    | 20.846  | 32.667      | 6          | TF        | fallen-river-715     |
| 256            | gpt2-medium | -          | 2.994    | 20.637  | 32.041      | 0          | TF        | morning-feather-696  |
| 512            | gpt2-medium | -          | 2.845    | 17.627  | 26.880      | 1          | TF        | crimson-elevator-716 |
| 1024           | gpt2-medium | -          | 2.758    | 15.927  | 24.035      | 2          | TF        | usual-violet-717     |
| 256            | gpt2-xl     | -          | 2.959    | 20.819  | 32.588      | 0          | TF        | crisp-dawn-723       |
| 512            | gpt2-xl     | -          | 2.683    | 15.318  | 22.978      | 0          | TF        | misty-plasma-722     |
| 1024           | gpt2-xl     | -          | 2.553    | 13.289  | 19.54       | 0          | TF        | woven-plant-721      |
