# Context length

https://app.wandb.ai/bkkaggle/lm-finetuning/reports/1-epoch-context-len--Vmlldzo3NTI4MA/edit

## Comments

-   larger context len is better
-   diminishing returns after gpt2-large
-   there is a 4 ppl difference between gpt2-medium and gpt2-large
-   larger difference between model sizes when context len is larger
-   prelim recommend to never finetune gpt2-xl, benefits are small

## Dataset

-   Wikitext-2

## Context Len

-   256
-   512
-   1024

## Hyperparameters

| model       | batch size | epochs | optimizer | learning rate |
| ----------- | ---------- | ------ | --------- | ------------- |
| gpt2-medium | 8          | 1      | Adafactor | 5e-5          |
| gpt2-large  | 8          | 1      | Adafactor | 5e-5          |
| gpt2-xl     | 8          | 1      | Adafactor | 5e-5          |

## Results

_Selected on best val score_

| context length | model       | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run                  |
| -------------- | ----------- | ---------- | -------- | ------- | ----------- | ---------- | --------- | -------------------- |
| 256            | gpt2-medium | -          | 2.971    | 20.875  | 32.641      | 0          | TF        | clear-puddle-793     |
| 512            | gpt2-medium | -          | 2.842    | 17.927  | 27.508      | 0          | TF        | pious-music-801      |
| 1024           | gpt2-medium | -          | 2.756    | 16.271  | 24.647      | 0          | TF        | exalted-dream-802    |
| 256            | gpt2-large  | -          | 2.902    | 19.56   | 30.321      | 0          | TF        | cerulean-sunset-794  |
| 512            | gpt2-large  | -          | 2.713    | 15.764  | 23.74       | 0          | TF        | magic-pyramid-803    |
| 1024           | gpt2-large  | -          | 2.609    | 14.052  | 20.833      | 0          | TF        | volcanic-silence-804 |
| 256            | gpt2-xl     | -          | 2.838    | 18.419  | 28.323      | 0          | TF        | morning-vortex-795   |
| 512            | gpt2-xl     | -          | 2.623    | 14.425  | 21.466      | 0          | TF        | skilled-breeze-806   |
| 1024           | gpt2-xl     | -          | 2.513    | 12.746  | 18.659      | 0          | TF        | vibrant-eon-807      |
