# Comparing Context Lengths

See how models trained on different context lengths aren't comparable. Train lms on some set context len, and evaluate on different context lens.

## Train on 1024, eval at lower

### Results

val_loss gets increasingly worse as eval context len decreases

### Hyperparameters

-   gpt2-medium
-   trained on wikitext2
-   trained using a context len of 1024

### Experiment

| eval context len | loss | val_ppl | adj_val_ppl |
| ---------------- | ---- | ------- | ----------- |
| 1024             | 2.75 | 15.75   | 23.59       |
| 512              | 2.84 | 17.19   | 26.19       |
| 256              | 2.96 | 19.38   | 30.05       |
| 128              | 3.13 | 22.97   | 36.53       |

## Train on 256, eval at higher

### Results

### Hyperparameters

-   gpt2-medium
-   trained on wikitext2
-   trained using a context len of 256

### Experiment

| eval context len | loss | val_ppl | adj_val_ppl |
| ---------------- | ---- | ------- | ----------- |
| 256              |
| 512              |
| 1024             |
