# GPT-2

-   Paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
-   Paperwithcode: https://paperswithcode.com/paper/language-models-are-unsupervised-multitask
-   Site: https://openai.com/blog/better-language-models/

## Hyperparameters

-   seqlen: 1024 tokens
-   batch size: 512

## Dataset

-   Webtext
    -   all reddit links with > 3 karma
    -   45M links
    -   dragnet and newspaper to extract data
    -   8 million documents and 40GB text
    -   removed all wikipedia documents
-   byte pair tokenization
    -   byte-level bpe
    -   prevent bpe from merging across char categories (A, 1, !)
-   perplexity calculation
    -   invertible detokenizers
-   gpt2 tokenization is with eos tokens at end of each document

## Model

-   vocab size: 50,257
-   Differences from original transformer
    -   Layer norm to input of each sub-block
    -   Additional Layer norm after the final self-attention block
    -   Scale weights of residual layers by 1 / sqrt(number of residual)
    -   context size from 512 -> 1024 tokens

## Notes

-   All models underfit
