# Results

## Model size

-   lower ppl with larger models, but you get diminishing returns after gpt2-medium

## Adafactor

-   Adafactor performs slightly worse that AdamW when beta_1 = 0

## Batch size

-   this is when using early stopping to stop training after the first epoch
    -   Larger batch size is better
    -   It trains faster and to a lower loss
    -   larger batch size is almost always better, but improves for more epochs than at a smaller batch size
    -   So far, this has only been tested for wikitext2

## Context size

-   improves significantly with increased context length
-   smaller models can rival larger models when context len is larger

## SGD

-   SGD varies a lot over multiple runs
-   using an lr-scheduler lets you get only a little higher (~1ppl higher) than adafactor
-   SGD uses slightly less memory and takes 10x longer

## Larger datasets

-   larger datasets need larger batch sizes

## IMDB

-   needs larger batch size of 64
-   can't get as low of a ppl

## 1 epochs

-   to train on small datasets, either:
    -   use a large batch size (256) with more epochs and early stopping after 1-2
    -   use a small batch size (4-16) with 1 epoch
-   you can train on small datasets in 1 epoch
-   reduce batch size to counteract to small number of iterations

## train lms on some set context len, and evaluate on different context lens.

-   eval on larger context len is almost as good as training on a larger context len
-   But improvements from training at larger context lens are greater when using larger models (at most 2ppl on gpt2-xl context-len 1024)
-   PPl goes down by 4 when eval at 1024 instead of 256 for all models

## Train Lms on diff context lens, evaluate on the same

-   The models perform almost identically, not sure there is much point of finetuning on a larger context len

## Non finetuned Lms

-   Non-finetuned LMs can't compete with finetuned lms on ppl and definitely can't transfer over the text style
