# Context length

https://app.wandb.ai/bkkaggle/lm-finetuning/reports/context-length--Vmlldzo3MTA4OA

## Comments

-   can't train gpt2-xl with a batch size of 16 on 8 cores, need to use a tpuv3-32 with a batch size of 32
-   gpt2 medium can still be pretty competitive with gpt2-xl

## Dataset

-   Wikitext-2

## Context Len

-   256
-   512
-   1024

## Hyperparameters

| model       | batch size | epochs | optimizer | learning rate |
| ----------- | ---------- | ------ | --------- | ------------- |
| gpt2        | 8          | 10     | Adafactor | 5e-5          |
| gpt2-medium | 8          | 10     | Adafactor | 5e-5          |
| gpt2-xl     | 8          | 10     | Adafactor | 5e-5          |

## Results

_Selected on best val score_

| context length | model       | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run                   |
| -------------- | ----------- | ---------- | -------- | ------- | ----------- | ---------- | --------- | --------------------- |
| 256            | gpt2        | -          | 3.305    | 29.248  | 48.024      | 0          | TF        | fearless-water-737    |
| 512            | gpt2        | -          | 3.159    | 24.781  | 39.898      | 1          | TF        | balmy-blaze-738       |
| 1024           | gpt2        | -          | 3.041    | 21.723  | 34.352      | 1          | TF        | dry-plasma-739        |
| 256            | gpt2-medium | -          | 3.018    | 21.892  | 34.472      | 0          | TF        | likely-shadow-740     |
| 512            | gpt2-medium | -          | 2.864    | 18.342  | 28.239      | 0          | TF        | laced-flower-741      |
| 1024           | gpt2-medium | -          | 2.764    | 16.431  | 24.931      | 1          | TF        | dandy-firefly-742     |
| 256            | gpt2-xl     | -          | 2.961    | 20.868  | 32.68       | 0          | TF        | northern-feather-743  |
| 512            | gpt2-xl     | -          | 2.683    | 15.315  | 22.967      | 0          | Tf        | laced-snow-745        |
| 1024           | gpt2-xl     | -          | 2.558    | 13.348  | 19.639      | 0          | Tf        | curious-firebrand-746 |
