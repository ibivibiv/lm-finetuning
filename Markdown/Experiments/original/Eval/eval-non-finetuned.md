# evaluating non-finetuned lms

## Results

Non-finetuned LMs can't compete with finetuned lms on ppl and definitely can't transfer over the text style

## Hyperparameters

-   original models are trained at 1024

### gpt2-medium

| eval context len | loss | val_ppl | adj_val_ppl |
| ---------------- | ---- | ------- | ----------- |
| 256              | 3.46 | 31.98   | 53.41       |
| 512              | 3.24 | 25.74   | 41.62       |
| 1024             | 3.08 | 21.77   | 34.34       |

### gpt2-large

| eval context len | loss | val_ppl | adj_val_ppl |
| ---------------- | ---- | ------- | ----------- |
| 256              | 3.33 | 27.97   | 45.80       |
| 512              | 3.10 | 22.30   | 35.31       |
| 1024             | 2.93 | 18.73   | 28.90       |

### gpt2-xl

| eval context len | loss | val_ppl | adj_val_ppl |
| ---------------- | ---- | ------- | ----------- |
| 256              | 3.22 | 25.22   | 40.67       |
| 512              | 3.00 | 20.12   | 31.37       |
| 1024             | 2.82 | 16.91   | 25.72       |

### Detokenizers + test set + seqlen 1024

| model       | loss  | val_ppl | adj_val_ppl | papers ppl wikitext103 |
| ----------- | ----- | ------- | ----------- | ---------------------- |
| gpt2        | 3.149 | 23.33   | 35.12       | 37.5                   |
| gpt2-medium | 2.923 | 18.59   | 27.18       | 26.37                  |
| gpt2-large  | 2.786 | 16.23   | 23.30       | 22.05                  |
| gpt2-xl     |
