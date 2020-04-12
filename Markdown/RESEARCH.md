# Research

## Notes

-   n_tokens and n_original_tokens not the same when using `--fast` or `--efficient`
-   adj ppl not accurate when using `--n_batches` or `--n_tokens`
-   use tokenized wikitext2?

## Objectives

-   If I wanted to finetune a LM to generate text of a specific style/content, what good defaults would I choose?

-   Find best practices or good defaults for finetuning tranformer language models for text generation.

-   Understand the effect of context len, model, and dataset size on generating coherent text
-   Understand the effect of loss and sampling stategies on generating coherent text

-   Understand the "theory" of how language models can be finetuned
-   The extent of language models needing to be large to generate coherent text

-   Present the smallest and most resource efficient LM that can generate coherent text
-   Present a finetuned LM that can generate coherent text across a range of domains
-   Finetune lms for a variety of tasks

-   publish LMs with transformers-cli
-   have a pytorch and TF codebase for easy finetuning
-   writeup the whole process

## ToDo

-   evaluation experiments
-   training experiments
-   new lm experiments

-   evaluate lms on the actual test set
-   use invertible detokenizers and word level data like in gpt2 to get ppl calculations

-   check other models
    -   check eval ppl code for
        -   tf xl
            -   uses word level pre-tokenized wikitext
        -   distilgpt2
            -   _not sure, check further_
        -   unlikelihood training
            -   word level pre-tokenized wikitext
        -   megatron lm
            -   megatron-lm normalizes the loss in my way (https://github.com/NVIDIA/Megatron-LM/blob/master/evaluate_gpt2.py#L282)
                -   uses word level wikitext data
                -   uses invertible tokenizers
                    -   https://github.com/NVIDIA/Megatron-LM/blob/master/detokenizer.py
                -   evals on sliding windows
                    -   https://github.com/NVIDIA/Megatron-LM/blob/master/evaluate_gpt2.py#L350
    -   _todo_:
        -   use word level
        -   use detokenizers
        -   use adjusted loss
        -   consider effect of sliding windows/eval seq len
-   run multiple times with different random seeds
-   make sure test set dataloader doesn't drop last batch
-   check if pytorch grad accumulation works similarly to tf
    -   then run gpt2-xl experiments on larger batch sizes

## New LMs

-   does finetuning give better results than grover, ctrl, etc
-   multiple datasets/control codes
    -   use control codes to train on a range of small (?) datasets
    -   is finetuning lm to work with control codes enough? or should it be pretrained with control codes too?
-   long-form generation with sliding windows
    -   Try leaking data to the lm by training on contigous sequences (move the start position of input sequence 1 step up
        -   would this even work or make a difference?
    -   can tf-xl be replaced with gpt2 and sliding windows
-   pplm
    -   generalize pplm
        -   more attribute models for normal generation use
        -   use a nn for the attribute model
            -   train to predict topics
            -   should make pplm edit text to make it more like chosen topic
-   albert style gpt2
-   distillation
-   train a transformer without layernorm
-   adapt ELECTRA to auto-regressive language modelling
-   finetune on multiple datasets with control codes if individual dataset is too small
-   train gpt2-xl on a larger context len
-   tf-xl is trained on a larger context len and isn't comparable

## Training

-   finetune on small-medium datasets (1M - 10M tokens)
-   use large context len
-   check if 2-3 epochs is better
    -   and if it is for larger (imdb) datasets
-   redo all with context len 1024
-   datasets
    -   should you train differently when dataset is:
        -   large
            -   wikitext103
            -   writingprompts
        -   small
            -   wikitext2
        -   specialized
            -   imdb
        -   long context _later_
            -   pg19
        -   show proof that it is/isn't
-   objective
    -   mle
    -   unlikelihood
    -   electra
-   fp16 _verify that it doesn't work_
    -   pure fp16 training for gpus
    -   tpu forced bfloat16
-   sgd training to fit on gpu _colab_
-   double descent
    -   Can lms where train loss is close to 0 generate text well?
    -   finetune for a long time, does double descent happen when train loss is 0?
    -   anything special about generating text that makes double descent not a good choice?
-   better to train on repeated parts of dataset, or go through a large dataset once
    -   try with a large dataset

## Evaluation

## Finetuning Datasets

Some language models might have been pretrained on some of these datasets.

-   classic lm datasets
    -   Wikitext
        -   from: https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/
        -   People train on raw data then normalize perplexity to match wikitext's tokenization
        -   Wikitext103
            -   ~550MB
            -   100M words
            -   1M sequences of length 256
            -   ~10min to process with normal
        -   Wikitext2
            -   ~10MB
            -   2M words
            -   10k sequences of length 256
            -   ~10s to process
-   reviews
    -   IMDB (Google Drive)
        -   https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
        -   https://ai.stanford.edu/~amaas/data/sentiment/
        -   50MB
        -   10M words
        -   50k sequences of length 256
-   books/stories
    -   WritingsPrompts (Heiarchichal neural story generation. Fan et al 2018)
        -   from: https://github.com/pytorch/fairseq/tree/master/examples/stories
        -   paper with metrics: https://arxiv.org/pdf/1805.04833.pdf
        -   paper models only on the first 1k words, but try more
        -   1 line/story
        -   comes with pairs of wp and story
        -   ~900MB
        -   200M words
        -   700k sequences of length 256
    -   PG-19 Language Modelling Benchmark
        -   https://github.com/deepmind/pg19
        -   Stored as a GCP bucket
        -   Train set is 10GB
        -   Val set
            -   50 books
            -   5M words
            -   20k sequences of length 256
        -   Test set
            -   100 books
            -   10M words
            -   40k sequences of length 256
-   news articles
    -   CNN/Daily Mail (Hermann et al, 2015)
        -   raw dataset from: https://cs.nyu.edu/~kcho/DMQA/
        -   Very large: 1.3GB
        -   300M words
        -   10M sequences of length 256
    -   CTRL data **get**

## Models

-   gpt2
-   ctrl
-   grover
-   electra
-   distilgpt2
-   dialogpt2
-   long context lens
    -   compressive transformer
    -   reformer
    -   transformer xl
    -   sparse transformer
    -   Adaptive Span Transformers
    -   Routing Transformer

## Done

-   model size
    -   bigger is better, but only until gpt2-medium
    -   verify that this holds with larger context lengths
        -   it does, larger models are always better, ~3ppl for gpt2-medium vs gpt2-xl
-   batch size
    -   bigger batch (64) is better
    -   bigger batch size is necessary and better as dataset size grows
-   context size
    -   bigger context len is better
    -   smaller models can rival larger models if their context len is larger
-   sampling strategy
    -   how to evaluate?
-   model type
-   optimizers
    -   sgd/momentum
        -   sgd gets similar results over a lot more epochs
        -   momentum isn't worth it b/c adafactor uses less memory than sgd + momentum
    -   adamw
        -   works the best but uses too much memory
    -   adafactor
        -   default
            -   works slightly worse than adamw but is needed for larger lms
        -   beta1
            -   no difference
        -   warmup and adafactor
            -   warmup is hard to replace, not worth it
-   see if biases and layernorm should be finetuned
    -   not finetuning bias and layernorm is the default for only AdamW
    -   but other implementations also use it for adafactor
-   all experiments so far are done with epochs = 10
    -   run wikitext2 with 1 epoch
-   redo all with epochs = 1
-   train lms on some set context len, and evaluate on different context lens.
-   see how models trained on different context lengths aren't comparable
    -   train lms on different context lens, and evaluate on set context len.
    -   larger context len is better
-   how much worse is text generated by a non-finetuned lm?
    -   run evaluation on non-finetuned lm
-   read through papers
    -   update md files
    -   gather notes
    -   see if gpt2-xl finetuning experiments are similar to papers
    -   take a look at memory saving techniques in the reformer
-   sampling
    -   does a decrease in ppl lead to better text?
        -   usually, yes
    -   any better evaluation metrics than ppl
        -   not that I know of
        -   pretty much everyone uses ppl
    -   see if model size changes anything about ppl
        -   larger models can have lower ppl
        -   sample different models
    -   see if trained/eval context len changes anything about text quality
        -   nope, not much
    -   generate with sliding windows
        -   done, no worse than normal text
-   based on a user's computation budget, should they finetune a small lm or use a large lm
    -   see other papers' opinions on this
    -   Scaling laws: pretrain a larger lm

### Won't do

-   enwiki8 (Looks like this has all the XML? or not)
    -   text8 (Cleaned version of enwiki8)
    -   progress measured in bpc, so probably won't use it
-   Yahoo answers
-   Project Gutenberg
    -   replaced by PG-19
    -   data from: https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html
-   Amazon reviews (McAuley et al, 2015)
    -   won't use
-   Bookscorpus (Aligning books and movies... Zhu et al, 2015)
    -   won't use
-   One billion words
    -   won't use
-   Yelp reviews (Character level convolutional networks... Zhang et al, 2015)
    -   won't use
-   CC-Stories (Trinh and Lee 2018)
    -   not relavent and can't find data
-   AG News
    -   only has title and descriptions
    -   https://course.fast.ai/datasets#nlp
-   OpenWebText (Megatron LM version)
    -   openwebtext repo
    -   newspaper to download text
    -   langdetect to filter content
    -   ftfy for unicode normalization
    -   filter out docs under 128 tokens
    -   lsh to deduplicate content with jaccard similarity more than 0.7
    -   end of text token to end of document
    -   174 GB of text
-   Wikipedia (Devlin et al 2018)
    -   wikitext is a good enough alternative
-   what is the best pretrained lm for text gen?
    -   best for widest range of text gen
    -   best for specific type of text gen (e.g. news, dialog)
-   https://github.com/deepmind/lamb ?
-   lars?
-   learning rate decay strategies
-   inv sqrt decay
-   effect of gpt-2's gelu and layernorm to inputs
-   try increasing context size over time
-   Penn Treebank
    -   get from https://github.com/salesforce/awd-lstm-lm/blob/master/getdata.sh#L33
    -   ~5MB
    -   1M words
    -   5k sequences of length 256
-   online comments
-   effect of tokenizing approach (line by line, seq_len length chunks, lazy loading with random start point)
    -   `--fast` doesn't work well with small lines, it gives out 3x more examples than normal for wikitext2
        -   also makes the result not entirely comparable with most other approaches
    -   `efficient` has loss at 1.x
    -   should have a negligble difference, see whats the most efficient and what the performance diff is
    -   what is the best way to tokenize
    -   have a range of ways for different dataset sizes
    -   lazy loading
        -   choose a random start point
        -   get 1k chars from that point
        -   tokenize and discard excess

### Resources

-   https://course.fast.ai/datasets
-   https://paperswithcode.com/task/language-modelling
