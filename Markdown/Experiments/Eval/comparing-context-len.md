# Comparing Context Lengths

See how models trained on different context lengths aren't comparable. Train lms on some set context len, and evaluate on different context lens.

## Hyperparameters

-   gpt2-medium
-   trained on wikitext2
-   temperature 1

## Train on 1024, eval at lower

### Results

val_loss gets increasingly worse as eval context len decreases

### Hyperparameters

-   trained using a context len of 1024

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
-   _check if text quality isn't related to ppl only_

### Hyperparameters

-   trained using a context len of 256

### Experiment

| eval context len | loss  | val_ppl | adj_val_ppl |
| ---------------- | ----- | ------- | ----------- |
| 256              | 2.971 | 20.875  | 32.641      |
| 512              | 2.85  | 17.44   | 26.64       |
| 1024             | 2.78  | 16.12   | 24.33       |
