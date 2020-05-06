# training

# datasets

- wikitext2
- wikitext103
- imdb

# models

-   gpt2 - 124M parameters
-   gpt2-medium - 355M parameters
-   gpt2-large - 774M parameters
-   gpt2-xl - 1.5B parameters

# hyperparameters

| epochs | optimizer | learning rate |
| ------ | --------- | ------------- |
| 1      | Adafactor | 5e-5          |

# experiments

| dataset   | context len | model   | batch size | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run                  |
| --------- | ----------- |-------- | ---------- | ---------- | -------- | ------- | ----------- | ---------- | --------- | -------------------- |
| wikitext2 | 256         | gpt2    | 8          |