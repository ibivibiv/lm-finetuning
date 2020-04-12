# Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

-   Paper: https://arxiv.org/abs/1901.02860
-   Paperswithcode: https://paperswithcode.com/paper/transformer-xl-attentive-language-models
-   Site: https://ai.googleblog.com/2019/01/transformer-xl-unleashing-potential-of.html
-   ResearchGate: https://openreview.net/forum?id=HJePno0cYm
-   Github: https://github.com/kimiyoung/transformer-xl

## Hyperparameters

-   seqlen: 384 in training and 1600 test
-   batch size: 512
-   iterations: 800k
-   optimizer: Adafactor
-   weight decay: 0.01
-   learning rate: 1e-4
-   warmup: 10k

## Dataset

-   wikitext103
    -   avg len of 3k tokens/article
    -   use pretokenized wikitext2 and 103

## Model

-   caches hidden state for previous timesteps and turns off grads for it
-   relative positional encodings to know relative difference between two vectors
-   151M and 257M sizes
    -   only 257M is available on /transformers
-   only tf checkpoints for 257M on original repository
    -   but available trained on multiple datasets

## Notes

-   only the official implementation has the correct training functionality
