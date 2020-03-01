# ToDo

-   find lm trained with unlikelihood objective
-   look at compressive transformer and reformer
-   upload saved models quickly
    -   wait for fix
-   pytorch lightning?
-   tf
    -   data
        -   tfrecords
    -   model
        -   huggingface/transformers
        -   how to disable training for certain layers
    -   training
        -   keras
    -   tpu
        -   looks like it doesn't work in colab
        -   error: https://github.com/tensorflow/tensorflow/issues/34391

# Done

-   look at citations for gpt2
-   Find papers
-   find repositories
-   Read papers
-   get a constant code format
-   summarize major contributions from papers
-   look at notes
-   https://github.com/deepmind/pg19
-   use tokenizers repo
-   deal with multiple files
-   customize/improve code
-   How to compute perplexity on specific datasets
-   download datasets
    -   Drive?
-   test model
-   check ppl calc
    -   ppl is correct
    -   check if adjusted ppl is correct
-   add flag to only use first n_batches to train
-   add flag to only consider first n tokens in file/lin

# Wont do

-   find colab notebooks on finetuning
-   annotate MASS
-   https://colab.research.google.com/drive/1-ROO7L09EupLFLQM-TWgDHa5-FIOdLLh
-   colab remote tool
    -   not worth it

# Later

-   find old papers
-   Find other guides and papers on how to finetune
-   table to compare papers
-   turing nlg - not available yet
-   think about data that's to large to fit in RAM later
-   tokenize on the fly?
-   TPU support: https://github.com/pytorch/xla/pull/1650
    -   mp still doesn't work
-   update wandb code
-   fix resuming
    -   has to wait until fasttokenizer is fixed
-   remote debugging
    -   wait until using vms
-   ctrl control codes
    -   prob redo dataset processing to save metadata
