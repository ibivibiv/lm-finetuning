# Model size

## Comments

PPL decreases with model size.
A model of a particular size cannot be trained to be better than a certain ppl.
Larger models need to be trained for significantly less epochs (1-2 on wikitext-2) than smaller models
Diminishing returns after gpt2-medium

## Recommendations

-   Finetune gpt2-medium since gpt2-xl can't be trained with adamw

## Dataset

-   Wikitext-2

## Model types

-   distilgpt2 - 82M parameters - max batch size 16
-   gpt2 - 124M parameters - max batch size 16
-   gpt2-medium - 355M parameters - max batch size 4
-   gpt2-large - 774M parameters - max batch size 1 - with grad checkpointing on GPU
-   gpt2-xl - 1.5B parameters

## Hyperparameters

| seq len | epochs | global batch size | optimizer | learning rate |
| ------- | ------ | ----------------- | --------- | ------------- |
| 256     | 10     | 16                | AdamW     | 5e-5          |

## Results

_Selected on best val score_

| model       | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run                 |
| ----------- | ---------- | -------- | ------- | ----------- | ---------- | --------- | ------------------- |
| distilgpt2  | -          | 3.486    | 32.662  | 54.715      | 2          | Pytorch   | rosy-wildflower-616 |
| gpt2        | -          | 3.23     | 25.273  | 40.76       | 0          | Pytorch   | silvery-galaxy-617  |
| gpt2-medium | -          | 2.959    | 19.288  | 29.888      | 0          | Pytorch   | ethereal-cosmos-619 |
| gpt2-large  | -          | 2.858    | 18.052  | 27.499      | 0          | TF        | divine-elevator-666 |
| gpt2-xl     | OOM        |
