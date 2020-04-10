# train at a set context len, eval at different context lens

See how models trained on different context lengths aren't comparable. Train lms on some set context len, and evaluate on different context lens.

## Results

eval on larger context len is almost as good as training on a larger context len

But improvements from training at larger context lens are greater when using larger models (at most 2ppl on gpt2-xl context-len 1024)

PPl goes down by 4 when eval at 1024 instead of 256 for all models

Improvement of training at 1024 is even less when training at 512 instead of 256

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
-   original differences between model sizes is at most 4ppl anyway
-   eval on larger context len is almost as good as training on a larger context len
-   But improvements from training at larger context lens are greater when using larger models (<1ppl for gpt2-medium and at most 2ppl on gpt2-xl)
-   PPl goes down by 4 when eval at 1024 instead of 256 for all models
    -   The ppl metric itself becomes that much easier for models that are of comparable performance at higher context lens

### Hyperparameters

-   trained using a context len of 256
-   gpt2-medium

### Experiment

| eval context len | loss  | val_ppl | adj_val_ppl |
| ---------------- | ----- | ------- | ----------- |
| 256              | 2.971 | 20.875  | 32.641      |
| 512              | 2.85  | 17.44   | 26.64       |
| 1024             | 2.78  | 16.12   | 24.33       |

### Original

| context length | model       | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run               |
| -------------- | ----------- | ---------- | -------- | ------- | ----------- | ---------- | --------- | ----------------- |
| 256            | gpt2-medium | -          | 2.971    | 20.875  | 32.641      | 0          | TF        | clear-puddle-793  |
| 512            | gpt2-medium | -          | 2.842    | 17.927  | 27.508      | 0          | TF        | pious-music-801   |
| 1024           | gpt2-medium | -          | 2.756    | 16.271  | 24.647      | 0          | TF        | exalted-dream-802 |

### gpt2-large

#### Results

-   ppl goes down with seqlen
-   there is a 1ppl diff at most, more than gpt2-medium

#### Hyperparameters

-   trained using a context len of 256
-   gpt2-large

#### Experiment

| eval context len | loss  | val_ppl | adj_val_ppl |
| ---------------- | ----- | ------- | ----------- |
| 256              | 2.902 | 19.56   | 30.321      |
| 512              | 2.79  | 16.29   | 24.62       |
| 1024             | 2.71  | 15.07   | 22.51       |

#### Original

| context length | model      | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run                  |
| -------------- | ---------- | ---------- | -------- | ------- | ----------- | ---------- | --------- | -------------------- |
| 256            | gpt2-large | -          | 2.902    | 19.56   | 30.321      | 0          | TF        | cerulean-sunset-794  |
| 512            | gpt2-large | -          | 2.713    | 15.764  | 23.74       | 0          | TF        | magic-pyramid-803    |
| 1024           | gpt2-large | -          | 2.609    | 14.052  | 20.833      | 0          | TF        | volcanic-silence-804 |

### gpt2-xl

#### Results

Same, eval on larger context len is almost as good as training on a larger context len

But improvements from training at larger context lens are greater when using larger models (at most 2ppl on gpt2-xl context-len 1024)

PPl goes down by 4 when eval at 1024 instead of 256 for all models

#### Hyperparameters

-   trained using a context len of 256
-   gpt2-xl

#### Experiment

| eval context len | loss  | val_ppl | adj_val_ppl |
| ---------------- | ----- | ------- | ----------- |
| 256              | 2.838 | 18.419  | 28.323      |
| 512              | 2.72  | 15.29   | 22.893      |
| 1024             | 2.65  | 14.17   | 20.98       |

#### Original

| context length | model   | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run                |
| -------------- | ------- | ---------- | -------- | ------- | ----------- | ---------- | --------- | ------------------ |
| 256            | gpt2-xl | -          | 2.838    | 18.419  | 28.323      | 0          | TF        | morning-vortex-795 |
| 512            | gpt2-xl | -          | 2.623    | 14.425  | 21.466      | 0          | TF        | skilled-breeze-806 |
| 1024           | gpt2-xl | -          | 2.513    | 12.746  | 18.659      | 0          | TF        | vibrant-eon-807    |

## Train on 512, eval at higher

### Results

-   There is less of a difference between eval at 512 and 1024 vs training and eval at 512 when the model is trained at 512 instead of 256

### gpt2-medium

#### results

-   practically no difference

#### hyperparameters

-   train at context len 512

#### experiment

| eval context len | loss  | val_ppl | adj_val_ppl |
| ---------------- | ----- | ------- | ----------- |
| 512              | 2.842 | 17.927  | 27.508      |
| 1024             | 2.75  | 15.78   | 23.74       |

#### original

| context length | model       | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run               |
| -------------- | ----------- | ---------- | -------- | ------- | ----------- | ---------- | --------- | ----------------- |
| 512            | gpt2-medium | -          | 2.842    | 17.927  | 27.508      | 0          | TF        | pious-music-801   |
| 1024           | gpt2-medium | -          | 2.756    | 16.271  | 24.647      | 0          | TF        | exalted-dream-802 |

### gpt2-large

#### results

-   less of a diff when eval at 1024 if trained at 512 than if trained at 256
    -   1ppl diff to almost 0

#### hyperparameters

-   train at context len 512

#### experiment

| eval context len | loss  | val_ppl | adj_val_ppl |
| ---------------- | ----- | ------- | ----------- |
| 512              | 2.713 | 15.764  | 23.74       |
| 1024             | 2.63  | 13.91   | 20.54       |

#### original

| context length | model      | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run                  |
| -------------- | ---------- | ---------- | -------- | ------- | ----------- | ---------- | --------- | -------------------- |
| 512            | gpt2-large | -          | 2.713    | 15.764  | 23.74       | 0          | TF        | magic-pyramid-803    |
| 1024           | gpt2-large | -          | 2.609    | 14.052  | 20.833      | 0          | TF        | volcanic-silence-804 |

### gpt2-xl

#### results

-   even less of a diff than gpt2-large

#### hyperparameters

#### experiment

| eval context len | loss  | val_ppl | adj_val_ppl |
| ---------------- | ----- | ------- | ----------- |
| 512              | 2.623 | 14.425  | 21.466      |
| 1024             | 2.54  | 12.74   | 18.56       |

#### original

| context length | model   | train loss | val loss | val ppl | adj val ppl | best epoch | framework | run                |
| -------------- | ------- | ---------- | -------- | ------- | ----------- | ---------- | --------- | ------------------ |
| 512            | gpt2-xl | -          | 2.623    | 14.425  | 21.466      | 0          | TF        | skilled-breeze-806 |
| 1024           | gpt2-xl | -          | 2.513    | 12.746  | 18.659      | 0          | TF        | vibrant-eon-807    |
