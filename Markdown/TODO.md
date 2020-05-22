# ToDo

-   keep track of everything to reproduce results
-   do evaluation experiments
    -   train
        -   model size
        -   train context len
        -   on wikitext2, wikitext103, and imdb
    -   non finetuned
    -   train at multiple context lengths, eval at the same
    -   train at a set context len, eval at different context lens

# info

-   pytorch lightning
    -   todo
        -   wandb support
            -   logging
                -   wait for fix for train loss and train ppl
                -   https://github.com/PyTorchLightning/pytorch-lightning/issues/1076
            -   watch
                -   where to put watch() call?
        -   grad checkpointing
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

# next time

-   distillation
    -   distilgpt2-xl
-   unlikelihood
    -   finetune?
-   no layernorm
-   control codes
-   pplm
-   evaluation context lens
-   double descent
-   use tensorboard more
-   dropout/inplace dropout

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
-   pytorch lightning
    -   done
        -   dataset
        -   untrainable params
        -   ppl
        -   val and test
        -   cli args
        -   check lr scaling
        -   check apex
        -   grad accumulation
        -   checkpointing
            -   save dir
            -   tokenizer checkpointing
                -   not necessary anymore
        -   early stopping
        -   lr schedule
            -   implement fix
        -   resuming
            -   pytorch lightning should handle optimizer and scheduler restoring
                -   wandb.run.dir
        -   waiting for fix for gpt2-large and precision=16
            -   fixed precision=16 myself
        -   go through docs
        -   fix grad clipping
        -   check test
            -   sampling
            -   eval_only flag
-   Get test/sampling code done to get on with research
-   try to get training code done ASAP
-   check if gpt2-xl can be trained with fp16
-   keras epoch checkpoint callback
-   disable grads for certain layers
    -   disabling grads for bias and layernorm is only used in the adamw paper
        -   https://github.com/huggingface/transformers/issues/492
-   get lms ready
    -   algpt2
-   reapply for tfrc
-   make datasets
    -   openwebtext
    -   pg19
    -   cnn/dailymail
    -   ctrl news dataset
-   retrain gpt2
-   train lms
-   finetune lms

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
-   https://cloud.google.com/tpu/docs/cloud-tpu-tools#op_profile
    -   new version
    -   setup up tensorboard on wandb
    -   wait until it gets native support in pt lightning and wandb
