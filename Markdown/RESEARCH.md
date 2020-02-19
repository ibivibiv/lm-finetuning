# Research

## Objectives

-   If I wanted to finetune a LM to generate text of a specific style/content, what good defaults would I choose?

-   Find best practices for finetuning tranformer language models for text generation.
-   Understand the "theory" of how language models can be finetuned
-   Present a finetuned LM that can generate coherent text across a range of domains
-   Understand the effect of model and dataset size on generating coherent text
-   Understand the effect of loss and sampling stategies on generating coherent text
-   Present the smallest and most resource efficient LM that can generate coherent text
-   The extent of language models needing to be large to generate coherent text

## Roadmap

-   Implement code to train LMs on different datasets
-   Start with a single dataset and a single transformer
-   See how text quality changes

## What to try

-   Try leaking data to the lm by training on contigous sequences (move the start position of input sequence 1 step up
-   context size
-   datasets
    -   large
    -   small
    -   custom
    -   well known
    -   specialized
    -   general
-   lm/pplm
-   loss
-   multiple datasets/control codes
-   evaluation metrics
-   perplexity?
-   other papers
-   dataset sampling strategy
-   optimizers
    -   adagrad
    -   adafactor
-   see if biases and layernorm should be finetuned
-   learning rate decay strategies
-   warmup
-   pure fp16 training
-   effect of gpt-2's gelu and layernorm to inputs
-   double descent
-   effect of tokenizing approach (line by line, seq_len length chunks)
    -   might have an effect on efficiency
-   Compare a small finetuned lm to a larger non-finetuned lm
-   Can lms where train loss is close to 0 generate text well?
-   based on a user's computation budget, should they finetune a small lm or use a large lm

## Finetuning Datasets

Some language models might have been pretrained on some of these datasets.

-   IMDB
-   AG News
-   Yahoo answers
-   Wikitext
    -   People train on raw data then normalize perplexity to match wikitext's tokenization
    -   Wikitext103
    -   Wikitext2
-   Penn Treebank
-   text8 (Cleaned version of enwiki8)
-   enwiki8 (Looks like this has all the XML? or not)
-   Project Gutenberg
-   Amazon reviews (McAuley et al, 2015)
-   CNN/Daily Mail (Hermann et al, 2015)
-   Bookscorpus (Aligning books and movies... Zhu et al, 2015)
-   One billion words
-   Yelp reviews (Character level convolutional networks... Zhang et al, 2015)
-   WritingsPrompts (Heiarchichal neural story generation. Fan et al 2018)
-   CC-Stories (Trinh and Lee 2018)
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
-   PG-19 Language Modelling Benchmark

### Resources

-   https://course.fast.ai/datasets
-   https://paperswithcode.com/task/language-modelling

## LMs
