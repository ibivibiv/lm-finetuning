# eval non finetuned

## setup

-   wikitext2 and 103 test sets
-   detokenizers
-   context len 1024

`python train_pt.py --val_path ./data/wikitext-2/wiki.test.tokens --seq_len 1024 --detokenizer --checkpoint distilgpt2 --batch_size 8 --eval_only`

## wikitext2

| model       | val loss | val ppl | adj val ppl |
| ----------- | -------- | ------- | ----------- |
| distilgpt2  | 3.44     | 31.27   | 48.71       |
| gpt2        | 3.14     | 23.15   | 34.70       |
| gpt2-medium | 2.90     | 18.24   | 26.5        |
| gpt2-large  | 2.77     | 16.00   | 22.87       |
| gpt2-xl     | 2.69     | 14.78   | 20.92       |

## wikitext103

| model       | val loss | val ppl | adj val ppl |
| ----------- | -------- | ------- | ----------- |
| distilgpt2  | 3.59     | 36.54   | 51.63       |
| gpt2        | 3.2573   | 25.97   | 35.51       |
| gpt2-medium | 2.96     | 19.32   | 25.68       |
| gpt2-large  | 2.81     | 16.69   | 21.88       |
| gpt2-xl     | 2.72     | 15.24   | 19.80       |
