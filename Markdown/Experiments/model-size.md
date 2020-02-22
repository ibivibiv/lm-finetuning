# Model size

## Model types

-   distilgpt2 - 82M parameters
-   gpt2 - 124M parameters
-   gpt2-medium - 355M parameters
-   gpt2-large - 774M parameters
-   gpt2-xl - 1.5B parameters

## Hyperparameters

| seq len | epochs | global batch size | optimizer | learning rate |
| ------- | ------ | ----------------- | --------- | ------------- |
| 256     | 10     | 16                | AdamW     | 5e-5          |

## Results

_Selected on best val score_

| model      | train loss | val loss | val ppl | adj val ppl |
| ---------- | ---------- | -------- | ------- | ----------- |
| distilgpt2 | 3.311      | 3.43     | 30.875  | 54.464      |
