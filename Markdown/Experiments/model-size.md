# Model size

## Dataset

-   Wikitext-2

## Model types

-   distilgpt2 - 82M parameters - max batch size 16
-   gpt2 - 124M parameters - max batch size 16
-   gpt2-medium - 355M parameters - max batch size 4
-   gpt2-large - 774M parameters
-   gpt2-xl - 1.5B parameters

## Hyperparameters

| seq len | epochs | global batch size | optimizer | learning rate |
| ------- | ------ | ----------------- | --------- | ------------- |
| 256     | 10     | 16                | AdamW     | 5e-5          |

## Results

_Selected on best val score_

| model       | train loss | val loss | val ppl | adj val ppl |
| ----------- | ---------- | -------- | ------- | ----------- |
| distilgpt2  | 3.311      | 3.43     | 30.875  | 54.464      |
| gpt2        | 3.094      | 3.186    | 24.192  | 40.987      |
| gpt2-medium | 2.757      | 2.915    | 18.449  | 29.885      |
| gpt2-large  |
