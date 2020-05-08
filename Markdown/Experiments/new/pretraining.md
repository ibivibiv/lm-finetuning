# Pretraining

## dataset

-   openwebtext
-   5% val split
-   40gb text
-   8m documents

## tokenization

-   byte level bpe
-   vocab size of 50257
-   ctrl uses a 5% split of data for training tokenizer

## preprocessing

-   add `<|endoftext|>` to beginning of each sequence

## model

-   seqlen 1024
-   start with gpt2-124M

## training

-   800k iterations
-   batch size: as high as possible
-   linear warmup for 10% of iterations (or 10k iterations?)
-   use adafactor with beta1 = 0
-   ~20 epochs
