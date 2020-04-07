# Comparing Context Lengths

See how models trained on different context lengths aren't comparable. Train lms on different context len, and evaluate on the same context len.

## Results

The models perform almost identically, not sure there is much point of finetuning on a larger context len

## Hyperparameters

-   trained on wikitext2
-   temperature 1
-   gpt2-medium

### eval at 256

#### Results

#### Experiment

| train context len | loss | val_ppl | adj_val_ppl | id                |
| ----------------- | ---- | ------- | ----------- | ----------------- |
| 256               | 2.97 | 19.51   | 30.29       | clear-puddle-793  |
| 512               | 2.95 | 19.24   | 29.81       | pious-music-801   |
| 1024              | 2.96 | 19.38   | 30.05       | exalted-dream-802 |

### eval at 512

#### Results

#### Experiment

| train context len | loss | val_ppl | adj_val_ppl | id                |
| ----------------- | ---- | ------- | ----------- | ----------------- |
| 256               | 2.85 | 17.45   | 26.64       | clear-puddle-793  |
| 512               | 2.84 | 17.14   | 26.10       | pious-music-801   |
| 1024              | 2.84 | 17.19   | 26.19       | exalted-dream-802 |
