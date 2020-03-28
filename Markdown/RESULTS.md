# Results

## Model size

-   lower ppl with larger models, but you get diminishing returns after gpt2-medium

## Adafactor

-   Adafactor performs slightly worse that AdamW when beta_1 = 0

## Batch size

-   Larger batch size is better
-   It trains faster and to a lower loss

## Context size

-   improves significantly with increased context length
-   smaller models can rival larger models when context len is larger
    -   _get data_
-   _note: gpt2-xl experiments were done with batch size 8 instead of 16_

## SGD

-   SGD varies a lot over multiple runs
-   using an lr-scheduler lets you get only a little higher (~1ppl higher) than adafactor
-   SGD uses slightly less memory and takes 10x longer
