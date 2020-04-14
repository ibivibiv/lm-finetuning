# train at multiple context lengths, eval at the same

See how models trained on different context lengths aren't comparable. Train lms on different context len, and evaluate on the same context len.

## Results

-   models that are trained on larger context lengths (> gpt2-medium) perform better on a given context len (2ppl, from 256->512; no big improvement for 1024)
    -   diminishing returns after gpt2-large
    -   training on a larger context len gives you a better language model
    -   but as expected, its not much

## Hyperparameters

-   trained on wikitext2
-   temperature 1

### gpt2-medium

#### eval at 256

##### Experiment

| train context len | loss | val_ppl | adj_val_ppl |
| ----------------- | ---- | ------- | ----------- |
| 256               | 2.97 | 19.51   | 30.29       |
| 512               | 2.95 | 19.24   | 29.81       |
| 1024              | 2.96 | 19.38   | 30.05       |

#### eval at 512

##### Experiment

| train context len | loss | val_ppl | adj_val_ppl |
| ----------------- | ---- | ------- | ----------- |
| 256               | 2.85 | 17.45   | 26.64       |
| 512               | 2.84 | 17.14   | 26.10       |
| 1024              | 2.84 | 17.19   | 26.19       |

#### eval at 1024

##### Experiment

| train context len | loss | val_ppl | adj_val_ppl |
| ----------------- | ---- | ------- | ----------- |
| 256               | 2.78 | 16.12   | 24.33       |
| 512               | 2.75 | 15.78   | 23.74       |
| 1024              | 2.75 | 15.75   | 23.69       |

### gpt2-large

#### eval at 256

##### Experiment

| train context len | loss | val_ppl | adj_val_ppl |
| ----------------- | ---- | ------- | ----------- |
| 256               | 2.90 | 18.20   | 27.97       |
| 512               | 2.79 | 16.29   | 24.62       |
| 1024              | 2.71 | 15.06   | 2.51        |

### gpt2-xl

#### eval at 256

##### Experiment

| train context len | loss | val_ppl | adj_val_ppl |
| ----------------- | ---- | ------- | ----------- |
| 256               | 2.83 | 17.09   | 26.01       |
| 512               | 2.73 | 15.44   | 23.15       |
| 1024              | 2.71 | 15.06   | 22.51       |

#### eval at 512

##### Experiment

| train context len | loss | val_ppl | adj_val_ppl |
| ----------------- | ---- | ------- | ----------- |
| 256               | 2.72 | 15.20   | 22.89       |
| 512               | 2.62 | 13.77   | 20.30       |
| 1024              | 2.59 | 13.42   | 19.71       |

#### eval at 1024

##### Experiment

| train context len | loss | val_ppl | adj_val_ppl |
| ----------------- | ---- | ------- | ----------- |
| 256               | 2.65 | 14.17   | 20.98       |
| 512               | 2.54 | 12.74   | 18.56       |
| 1024              | 2.51 | 12.36   | 17.94       |
