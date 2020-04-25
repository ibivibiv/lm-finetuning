# wikitext2

## hyperparameters

| seq len | epochs | global batch size | optimizer | learning rate |
| ------- | ------ | ----------------- | --------- | ------------- |
| 256     | 10     | 8                 | adafactor | 5e-5          |

## Results

| model | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run           |
| ----- | ---------- | -------- | ------- | ----------- | ---------- | --------- | ------------- |
| gpt2  | 5.85       | 5.34     | 208     | -           | 9          | tf        | deft-fire-978 |
