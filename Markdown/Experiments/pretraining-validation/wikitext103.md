# wikitext103

## notes

-   this is with `<|endoftext|>` added to the beginning of each sequence

## hyperparameters

| seq len | epochs | global batch size | optimizer | learning rate |
| ------- | ------ | ----------------- | --------- | ------------- |
| 256     | 10     | 64                | adafactor | 5e-5          |

## Results

| model | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run            |
| ----- | ---------- | -------- | ------- | ----------- | ---------- | --------- | -------------- |
| gpt2  | 5.23       | 4.96     | 142     | -           | 9          | TF        | hardy-haze-979 |
