# Comparing Context Lengths

See how models trained on different context lengths aren't comparable. Train lms on some set context len, and evaluate on different context lens.

## Hyperparameters

-   trained on wikitext2
-   temperature 1

## Train on 1024, eval at lower

### Results

val_loss gets increasingly worse as eval context len decreases

-   do lms trained on smaller context lens have more repeated n-grams
    -   count n-grams to determine actual number

### Hyperparameters

-   trained using a context len of 1024
-   gpt2-medium

### Experiment

| eval context len | loss | val_ppl | adj_val_ppl |
| ---------------- | ---- | ------- | ----------- |
| 128              | 3.13 | 22.97   | 36.53       |
| 256              | 2.96 | 19.38   | 30.05       |
| 512              | 2.84 | 17.19   | 26.19       |
| 1024             | 2.75 | 15.75   | 23.59       |

## Train on 256, eval at higher

### Results

-   ppl almost equal to that of larger models can be achieved by just increasing the context len at test time
-   makes it hard to differentiate between models of different sizes
-   _check if this stays for larger models_

### Hyperparameters

-   trained using a context len of 256
-   gpt2-medium

### Experiment

| eval context len | loss  | val_ppl | adj_val_ppl |
| ---------------- | ----- | ------- | ----------- |
| 256              | 2.971 | 20.875  | 32.641      |
| 512              | 2.85  | 17.44   | 26.64       |
| 1024             | 2.78  | 16.12   | 24.33       |

## gpt2-large

### Results

-   ppl goes down with seqlen

### Hyperparameters

-   trained using a context len of 256
-   gpt2-large

### Experiment

| eval context len | loss  | val_ppl | adj_val_ppl |
| ---------------- | ----- | ------- | ----------- |
| 256              | 2.902 | 19.56   | 30.321      |
| 512              | 2.79  | 16.29   | 24.62       |
| 1024             | 2.71  | 15.07   | 22.51       |

## gpt2-xl

### Results

### Hyperparameters

-   trained using a context len of 256
-   gpt2-xl

-   _compare against models trained at higher context lens_

### Experiment

| eval context len | loss  | val_ppl | adj_val_ppl |
| ---------------- | ----- | ------- | ----------- |
| 256              | 2.838 | 18.419  | 28.323      |
| 512              | 2.72  | 15.29   | 22.893      |
| 1024             |
