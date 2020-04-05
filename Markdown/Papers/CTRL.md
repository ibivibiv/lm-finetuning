# CTRL: A Conditional Transformer Language Model for Controllable Generation

-   Paper: https://arxiv.org/abs/1909.05858
-   Paperswithcode: https://paperswithcode.com/paper/ctrl-a-conditional-transformer-language-model
-   Site: https://blog.einstein.ai/introducing-a-conditional-transformer-language-model-for-controllable-generation/
-   Github: https://github.com/salesforce/ctrl

## Hyperparameters

-   seqlen: both 256 and 512 are available, only 512 is on /transformers
-   batch size: 1024 over 256 tpu cores, so 4 seq/core
-   iterations: 800k
-   optimizer: Adagrad
-   learning rate: 0.05
-   lr schedule: warmup 25k steps
-   no lr decay

## Dataset

-   full list: Appendix A, table 7
-   ~140GB of data
-   BPE tokenization with fastbpe
    -   english wikipedia and 5% split of OpenWebText for learning bpe tokens
-   filter out sequences with more thatn 2 unknown tokens
-   single stream of tokens with non-domain control codes at doc boundries
-   each sequence gets its control code prepended to it

## Model

-   vocab size: 250k

## Notes

-   little difference in quality in first 256 tokens when trained using 256 or 512 tokens
    -   possible b/c of 4x size of dataset and vocab
-   penalized sampling
    -   theta: 1.2
