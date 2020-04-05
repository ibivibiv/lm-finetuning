# Neural Text Generation with Unlikelihood Training

-   Arxiv: https://arxiv.org/abs/1908.04319
-   Paperswithcode: https://paperswithcode.com/paper/neural-text-generation-with-unlikelihood
-   OpenReview: https://openreview.net/forum?id=SJeYe0NtvH
-   Github: https://github.com/facebookresearch/unlikelihood_training

## Hyperparameters

-   seqlen: 1536
-   150k updates

## Dataset

-   wikitext103
-   fixed length contigous sequences

## Model

-   16 layer transformer

## Finetuning

-   1500 updates
-   50-50 chance to use mle vs unlikelihood
-   greedy decoding

## Notes

-   token level
    -   minimize the likelihood of tokens from the context from being generated again.
-   seq level
    -   penalize repeated n-grams on a decoded continuation
-   8 gpus
-   choose the checkpoint with the best val ppl
-   reduces the number of repeated sequences by a lot
