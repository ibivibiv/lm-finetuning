# Results

-   usually larger gap in context len performances when using a larger model

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

## train at a set context len, eval at different context lens

-   ppl almost equal to that of larger models can be achieved by just increasing the context len at test time
-   original differences between model sizes is at most 4ppl anyway
-   eval on larger context len is almost as good as training on a larger context len
-   But improvements from training at larger context lens are greater when using larger models (<1ppl for gpt2-medium and at most 2ppl on gpt2-xl)
-   PPl goes down by 4 when eval at 1024 instead of 256 for all models
    -   The ppl metric itself becomes that much easier for models that are of comparable performance at higher context lens
-   Improvement of training at 1024 is even less when training at 512 instead of 256

## train at multiple context lengths, eval at the same

-   models that are trained on larger context lengths (> gpt2-medium) perform better on a given context len

## evaluating non-finetuned lms

-   gpt2-medium sees almost no improvements
-   models that are trained on larger context lengths (> gpt2-medium) perform better on a given context len (2ppl, from 256->512; no big improvement for 1024)
    -   diminishing returns after gpt2-large
    -   training on a larger context len gives you a better language model
    -   but as expected, its not much

## Sampling, varying model size

-   No obvious differences in text quality, but can't be 100% sure

## sampling, train on 256, eval on multiple

-   No degredation of text quality after the first 256 chars

## Samping train 1024 eval context length

-   gpt2-xl looks only a little better possibly
