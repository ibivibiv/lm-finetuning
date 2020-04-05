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
