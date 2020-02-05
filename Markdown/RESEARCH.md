# Research

## Objectives

-   Find best practices for finetuning tranformer language models for text generation.
-   Understand the "theory" of how language models can be finetuned
-   Present a finetuned LM that can generate coherent text across a range of domains
-   Understand the effect of model and dataset size on generating coherent text
-   Understand the effect of loss and sampling stategies on generating coherent text
-   Present the smallest and most resource efficient LM that can generate coherent text
-   The extent of language models needing to be large to generate coherent text

## What to try

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

## Finetuning Datasets

Some language models might have been pretrained on some of these datasets.

-   IMDB
-   AG News
-   Yahoo answers
-   Wikitext103
-   wikitext2
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

### Resources

-   https://course.fast.ai/datasets
-   https://paperswithcode.com/task/language-modelling

## LMs
