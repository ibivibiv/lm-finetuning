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
-   use `--min_seq_len` (have to)
-   seqlen 1024 means that ~67% of files are too small
-   seqlen 512 means that ~35% of files are too small
-   seqlen 256 means that ~10% of files are too small
-   eventually concat files with smaller seqlens together
    -   done
-   approach:
    -   docs smaller than seqlen get concated together with eos token prepended to each
    -   eos token gets added at beginning of sequence

## model

-   seqlen 1024
-   start with gpt2-124M

## training

-   800k iterations
-   batch size: as high as possible
-   linear warmup for 10% of iterations (or 10k iterations?)
-   use adafactor with beta1 = 0
-   ~20 epochs
