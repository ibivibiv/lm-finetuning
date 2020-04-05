# Grover - Defending Against Neural Fake News

-   Paper: https://arxiv.org/abs/1905.12616
-   Paperswithcode: https://paperswithcode.com/paper/defending-against-neural-fake-news
-   Site: https://rowanzellers.com/grover/
-   Site: https://thegradient.pub/why-we-released-grover/
-   Github: https://github.com/rowanz/grover

## Hyperparameters

-   seqlen: 1024
-   batch size: 512
-   iterations: 800k
-   optimizer: Adafactor
-   weight decay: 0.01
-   learning rate: 1e-4
-   warmup: 10k

## Dataset

-   RealNews
    -   scraping data from 5000 news domains from common crawl
    -   used newspaper
    -   120 GB

## Model

-   same as gpt2

## Notes

-   256 tpu cores for two weeks
