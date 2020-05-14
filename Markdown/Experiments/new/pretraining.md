# Pretraining

## algpt2

-   just increasing the number of layers doesn't help almost at all
-   increasing the factorized embedding size helps more but its still no where close to as good

## notes

-   algpt2-124m doesn't really go below 4.5 loss
    -   but only tried ~60k iterations so far. more might be needed
    -   gpt2-124 should get below 3
-   factorized embedding size to #params
    -   128: 6.6e6
    -   256: 13e6
    -   512: 26e6
    -   1024: 53e6
    -   none: 80e6

## todo

-   do val on separate server

## dataset

-   openwebtext
-   5% val split
-   40gb text
-   8m documents

## tokenization

-   byte level bpe
-   vocab size of 50257
-   ctrl uses a 5% split of data for training tokenizer
-   i trained tokenizer on 1m files
-   same tokenizer for gpt2 and algpt2

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
-   train set
    -   53gb
    -   26gb of tfrecords
    -   7913604 files
    -   split into 16 tfrecords
    -   6885495 examples
-   val set
    -   339mb
    -   99998 files
    -   split into 16 tfrecords
    -   87011 examples

## model

-   seqlen 1024
-   start with gpt2-124M
-   then also pretrain algpt2-124m

## training

-   800k iterations
-   batch size: 512
-   linear warmup for 10k iterations
-   use adafactor with beta1 = 0
-   60 epochs; ~800k iterations

```
python3 train_tfrecords.py --tpu algpt2pod --seq_len 1024 --batch_size 512 --train_len 6885495 --warmup_steps 10000 --model_type algpt2 --config_path ./algpt2/algpt2/ --epochs 60 --train_path gs://algpt2/train/0.tfrecord gs://algpt2/train/1.tfrecord gs://algpt2/train/2.tfrecord gs://algpt2/train/3.tfrecord gs://algpt2/train/4.tfrecord gs://algpt2/train/5.tfrecord gs://algpt2/train/6.tfrecord gs://algpt2/train/7.tfrecord gs://algpt2/train/8.tfrecord gs://algpt2/train/9.tfrecord gs://algpt2/train/10.tfrecord gs://algpt2/train/11.tfrecord gs://algpt2/train/12.tfrecord gs://algpt2/train/13.tfrecord gs://algpt2/train/14.tfrecord gs://algpt2/train/15.tfrecord --val_path gs://algpt2/val/0.tfrecord gs://algpt2/val/1.tfrecord gs://algpt2/val/2.tfrecord gs://algpt2/val/3.tfrecord gs://algpt2/val/4.tfrecord gs://algpt2/val/5.tfrecord gs://algpt2/val/6.tfrecord gs://algpt2/val/7.tfrecord gs://algpt2/val/8.tfrecord gs://algpt2/val/9.tfrecord gs://algpt2/val/10.tfrecord gs://algpt2/val/11.tfrecord gs://algpt2/val/12.tfrecord gs://algpt2/val/13.tfrecord gs://algpt2/val/14.tfrecord gs://algpt2/val/15.tfrecord
```
