# Research

## Notes

-   don't use `--fast`, `--efficient`, `--n_batches`, or `--n_tokens` if you want correct results
    -   `--n_batches` doesn't work at all
-   tf drops last batch on training and its val metrics arent accurate
-   comments with `#check`
-   if multiple files, there must be a control code for each and an equal number of validation files
    -   val metrics will be averaged over training metrics

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

### Context len research

-   diminishing returns for increasing model size and context length
-   context len can decrease ppl values by itself

## ToDo

-   do gcp benchmarks
-   get tfrecords for tf version

    -   put tfrecords on gcp buckets
    -   expand callback and checkpoint

-   will see how much data is needed for pretraining
    -   tf-xl did it with only wikitext103
    -   gpt2 and variants used giant datasets
    -   will see if giant datasets are necessary
        -   giant datasets are more varied
    -   will need to use tfrecords or on the fly tokenization for giant datasets
        -   full dataset can't fit in memory
-   check ram needed for datasets
    -   wikitext103
        -   basic, _check on gcp_
        -   --fast, 13m
        -   --efficient, 10m
    -   benchmark on gcp
    -   will have to use --fast and --efficient for larger finetuning datasets
    -   get --fast working for tf
-   get framework ready for quickly running large scale experiments then reapply for tfrc
-   evaluation experiments
    -   redo wikitext2 and imdb experiments
        -   use word level
        -   run multiple times with different random seeds
        -   evaluate lms on the actual test set
-   training experiments
-   new lm experiments

## New LMs

-   datasets
    -   pg-19
        -   for large datasets, train for n iterations/batches instead of epochs
    -   wikitext103
    -   imdb
    -   writingprompts
    -   cnn/dailymail
-   new lms
    -   albert-style
    -   distilled
    -   no layernorm
        -   fixup init has been used before
        -   try in-place layer norm?
    -   electra
    -   unlikelihood
    -   double descent
    -   bert-style, but on webtext
-   ideas

    -   does finetuning give better results than grover, ctrl, etc
        -   finetune on news article datasets
    -   multiple datasets/control codes
        -   is finetuning lm to work with control codes enough? or should it be pretrained with control codes too?
        -   finetune on multiple datasets with control codes if individual dataset is too small
        -   use control codes to train on a range of small (?) datasets
            -   wikitext2
            -   imdb
        -   eval control code lm on only one dataset
            -   is it worse?
    -   just to prove that larger context lens make results uncomparable
        -   train gpt2-xl on a larger context len
        -   tf-xl is trained on a larger context len and isn't comparable

-   long-form generation with sliding windows

    -   Try leaking data to the lm by training on contigous sequences (move the start position of input sequence 1 step up
        -   would this even work or make a difference?
    -   can tf-xl be replaced with gpt2 and sliding windows
    -   eval tf-xl on smaller context lengths
        -   _is its extended context the reason for low ppl?_

-   pplm
    -   generalize pplm
        -   more attribute models for normal generation use
        -   use a nn for the attribute model
            -   train to predict topics
            -   should make pplm edit text to make it more like chosen topic

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
-   check other models
    -   check eval ppl code for
        -   gpt2
            -   uses word-level
        -   tf xl
            -   uses word level pre-tokenized wikitext
        -   distilgpt2
            -   not adjusted
        -   unlikelihood training
            -   word level pre-tokenized wikitext
        -   megatron lm
            -   megatron-lm normalizes the loss in my way (https://github.com/NVIDIA/Megatron-LM/blob/master/evaluate_gpt2.py#L282)
                -   uses word level wikitext data
                -   uses invertible tokenizers
                    -   https://github.com/NVIDIA/Megatron-LM/blob/master/detokenizer.py
                -   evals on sliding windows
                    -   https://github.com/NVIDIA/Megatron-LM/blob/master/evaluate_gpt2.py#L350
    -   gpt2 wikitext ppl numbers are a mess
        -   gpt2 uses word level data
        -   can't reproduce zero-shot results
            -   code for evaluation isn't available
            -   most likely possibility: they are reporting un-normalized results
                -   very close to my un-normalized zero-shot results too
            -   possible that they run the entire test set/each article in one context length then normalize
                -   would be like eval at a very high context len
        -   megatron-lm normalizes the loss in my way (https://github.com/NVIDIA/Megatron-LM/blob/master/evaluate_gpt2.py#L282)
            -   uses word level wikitext data
            -   uses invertible tokenizers
                -   https://github.com/NVIDIA/Megatron-LM/blob/master/detokenizer.py
            -   evals on sliding windows
                -   https://github.com/NVIDIA/Megatron-LM/blob/master/evaluate_gpt2.py#L350
        -   proof:
            -   https://github.com/huggingface/transformers/issues/483
            -   https://github.com/openai/gpt-2/issues/78
            -   https://github.com/huggingface/transformers/issues/491
            -   https://github.com/openai/gpt-2/issues/131
            -   https://github.com/openai/gpt-2/issues/131
-   make sure test set dataloader doesn't drop last batch
-   use detokenizers
-   check if pytorch grad accumulation works similarly to tf
    -   then run gpt2-xl experiments on larger batch sizes
-   tf doesn't work on colab
    -   need to roll back to tf2.1- control codes and multiple datasets
    -   tokenizer doesn't add eos token by default
    -   multiple datasets works
    -   for control codes
        -   prepend to each sequence
        -   can either use special tokens or just let it get tokenized
            -   will hardcore
        -   run temp experiments with wikitext2 and imdb
    -   make sure --fast and --efficient work
        -   they work
        -   both return 3x the number of sequences
        -   loss is too low because of padding
            -   masking isn't worth it, just use more ram and the tf version
    -   can't use more than the default control code with tf
-   add option to skip lines with less than seqlen tokens
    -   reduces wikitext103 train seqs from 1.16M to 140k
        -   100m tokens to 22m tokens
            -   won't work for wikitext, but it will for larger datasets
    -   also fixes the padding problem of low losses

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
