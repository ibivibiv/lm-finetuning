# Papers

### Name

#### Hyperparameters

-   seqlen:
-   batch size:
-   iterations:
-   optimizer:
-   weight decay:
-   gradient clipping:
-   learning rate:
-   warmup:
-   lr schedule:
-   dropout:
-   stop decay at lr

#### Dataset

-   byte pair tokenization
-   train-val ratio
-   token portions
-   perplexity calculation

#### Model

-   vocab size:
-   attention heads / layer:
-   layers:
-   model dimension:
-   feedforward dimension:

#### Notes

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

**Table of Contents** _generated with [DocToc](https://github.com/thlorenz/doctoc)_

-   [Papers](#papers)
    -   [Transformers](#transformers)
        -   [GPT-2](#gpt-2)
        -   [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](#megatron-lm-training-multi-billion-parameter-language-models-using-model-parallelism)
        -   [CTRL: A Conditional Transformer Language Model for Controllable Generation](#ctrl-a-conditional-transformer-language-model-for-controllable-generation)
        -   [T5 - Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](#t5---exploring-the-limits-of-transfer-learning-with-a-unified-text-to-text-transformer)
        -   [Grover - Defending Against Neural Fake News](#grover---defending-against-neural-fake-news)
        -   [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](#transformer-xl-attentive-language-models-beyond-a-fixed-length-context)
        -   [KERMIT: Generative Insertion-Based Modeling for Sequences](#kermit-generative-insertion-based-modeling-for-sequences)
        -   [Plug and Play Language Models: a Simple Approach to Controlled Text Generation](#plug-and-play-language-models-a-simple-approach-to-controlled-text-generation)
        -   [XLNet: Generalized Autoregressive Pretraining for Language Understanding](#xlnet-generalized-autoregressive-pretraining-for-language-understanding)
    -   [LM applied to areas](#lm-applied-to-areas)
        -   [DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](#dialogpt-large-scale-generative-pre-training-for-conversational-response-generation)
        -   [DLGNet: A Transformer-based Model for Dialogue Response Generation](#dlgnet-a-transformer-based-model-for-dialogue-response-generation)
        -   [A Pre-training Based Personalized Dialogue Generation Model with Persona-sparse Data](#a-pre-training-based-personalized-dialogue-generation-model-with-persona-sparse-data)
        -   [Generating Sentiment-Preserving Fake Online Reviews Using Neural Language Models and Their Human- and Machine-based Detection](#generating-sentiment-preserving-fake-online-reviews-using-neural-language-models-and-their-human--and-machine-based-detection)
        -   [Automated Speech Generation from UN General Assembly Statements: Mapping Risks in AI Generated Texts](#automated-speech-generation-from-un-general-assembly-statements-mapping-risks-in-ai-generated-texts)
        -   [Language models and Automated Essay Scoring](#language-models-and-automated-essay-scoring)
        -   [Read, Attend and Comment: A Deep Architecture for Automatic News Comment Generation](#read-attend-and-comment-a-deep-architecture-for-automatic-news-comment-generation)
        -   [Generating More Interesting Responses in Neural Conversation Models with Distributional Constraints](#generating-more-interesting-responses-in-neural-conversation-models-with-distributional-constraints)
        -   [Towards Controllable Story Generation](#towards-controllable-story-generation)
        -   [The Book of Endless History: Authorial Use of GPT2 for Interactive Storytelling](#the-book-of-endless-history-authorial-use-of-gpt2-for-interactive-storytelling)
        -   [Transfer Learning from Transformers to Fake News Challenge Stance Detection (FNC-1) Task](#transfer-learning-from-transformers-to-fake-news-challenge-stance-detection-fnc-1-task)
        -   [Zero-Shot Paraphrase Generation with Multilingual Language Models](#zero-shot-paraphrase-generation-with-multilingual-language-models)
    -   [Training objectives](#training-objectives)
        -   [Neural Text Generation with Unlikelihood Training](#neural-text-generation-with-unlikelihood-training)
    -   [Controlled generation/style tranfer](#controlled-generationstyle-tranfer)
        -   [XL-Editor: Post-editing Sentences with XLNet](#xl-editor-post-editing-sentences-with-xlnet)
        -   [Controlling Output Length in Neural Encoder-Decoders](#controlling-output-length-in-neural-encoder-decoders)
        -   [Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer](#delete-retrieve-generate-a-simple-approach-to-sentiment-and-style-transfer)
        -   [Controlling Linguistic Style Aspects in Neural Language Generation](#controlling-linguistic-style-aspects-in-neural-language-generation)
        -   [Controllable Text Generation](#controllable-text-generation)
        -   [Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation](#style-transformer-unpaired-text-style-transfer-without-disentangled-latent-representation)
        -   [Learning to Control the Fine-grained Sentiment for Story Ending Generation](#learning-to-control-the-fine-grained-sentiment-for-story-ending-generation)
        -   [Multiple-Attribute Text Rewriting](#multiple-attribute-text-rewriting)
        -   [Plan-And-Write: Towards Better Automatic Storytelling](#plan-and-write-towards-better-automatic-storytelling)
    -   [Discrimination](#discrimination)
        -   [The Detection of Distributional Discrepancy for Text Generation](#the-detection-of-distributional-discrepancy-for-text-generation)
        -   [Real or Fake? Learning to Discriminate Machine from Human Generated Text](#real-or-fake-learning-to-discriminate-machine-from-human-generated-text)
    -   [Sampling](#sampling)
        -   [The Curious Case of Neural Text Degeneration](#the-curious-case-of-neural-text-degeneration)
    -   [Evaluation](#evaluation)
        -   [Dynamic Evaluation of Transformer Language Models](#dynamic-evaluation-of-transformer-language-models)
        -   [How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation](#how-not-to-evaluate-your-dialogue-system-an-empirical-study-of-unsupervised-evaluation-metrics-for-dialogue-response-generation)
    -   [Other](#other)
        -   [Neural Text Generation: Past, Present and Beyond](#neural-text-generation-past-present-and-beyond)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Transformers

### Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

-   Paper: https://arxiv.org/abs/1909.08053
-   Paperwithcode: https://paperswithcode.com/paper/megatron-lm-training-multi-billion-parameter
-   Site: https://nv-adlr.github.io/MegatronLM
-   Github: https://github.com/NVIDIA/Megatron-LM

#### Hyperparameters

-   seqlen: 1024 subword units
-   batch size: 512
-   iterations: 300k
-   optimizer: Adam
-   weight decay: 0.01
-   gradient clipping: 1
-   learning rate: 1.5e-4
-   warmup: 3000 iterations (1%)
-   lr schedule: cosine linear decay
-   dropout: 0.1
-   stop decay at lr 1e-5

#### Dataset

-   Combining Wikipedia (Devlin et al 2018), RealNews, and OpenWebText
-   byte pair tokenization
-   29:1 train-val ratio
-   1024 token portions
-   wikitext103 perplexity calculation: see section 4.2.1

#### Model

-   attention head size: 96
-   vocab size: 50257
-   num heads and layers varied

#### Notes

-   uses GELU nonlinearities and layer norm to the inputs to distinct it from the transformer
-   layernorm params are duplicated to each gpu
-   perplexity decreases with model size: see section 5.2
-   possible future work:
    -   pretraining different model families
    -   different downstream tasks
    -   knowledge distillation

### T5 - Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

-   Paper: https://arxiv.org/abs/1910.10683
-   Paperswithcode: https://paperswithcode.com/paper/exploring-the-limits-of-transfer-learning
-   Github: https://github.com/google-research/text-to-text-transfer-transformer

#### Hyperparameters

-   seqlen: 512
-   batch size: 128
-   iterations: ~500k
-   optimizer: Adafactor
-   weight decay:
-   gradient clipping:
-   learning rate: 1e-2
-   warmup: 10^4 steps
-   lr schedule: inverse square root
-   dropout:
-   exponential decay until training ends
-   finetuning
    -   iterations: 250k
    -   seqlen: 128
    -   batch size: 512 (2^16 tokens/batch)
    -   learning rate: 1e-3

#### Dataset

-   Colossal Clean Crawled Corpus (C4)

    -   From Common Crawl April 2019
    -   cleaned using heuristics (see Section 2.2)
    -   langdetect to filter out non english text (0.99)
    -   750 GB of data

-   SentencePiece to encode data as wordpiece tokens
-   train-val ratio
-   token portions
-   perplexity calculation

#### Model

-   Encoder-decoder Vaswani

-   Baseline - 220M params - ~BERT base
    -   vocab size: 32k wordpieces
    -   attention heads / layer: 12
    -   layers: 12 (encoder and decoder each)
    -   model dimension: 768
    -   feedforward dimension: 3072
    -   dropout: 0.1

#### Notes

-   simplified positional encoding (see section 2.1)
-   Trained on 1024 TPUs with mesh tensorflow
-   Trained using maximum liklihood
    -   Teacher forcing
-   Task specific prefix to input
-   All tasks use greedy decoding
-   Pack multiple sequences into batch to fill 2^16 tokens/batch
-   Total tokens trained on is ~34B
-   BERT was pretrained on a total of 137B tokens and RoBERTa had 2.2T tokens
-   never repeat data during training
-   save checkpoints every 5k steps
-   SentencePiece
    -   trained on 10 parts english and 1 part german french or romanian

### KERMIT: Generative Insertion-Based Modeling for Sequences

-   Paper: https://arxiv.org/abs/1906.01604
-   Paperswithcode: https://paperswithcode.com/paper/kermit-generative-insertion-based-modeling
-   ResearchGate: https://www.researchgate.net/publication/333617133_KERMIT_Generative_Insertion-Based_Modeling_for_Sequences

### Plug and Play Language Models: a Simple Approach to Controlled Text Generation

-   Paper: https://arxiv.org/abs/1912.02164
-   Paperwithcode: https://paperswithcode.com/paper/plug-and-play-language-models-a-simple
-   Site: https://eng.uber.com/pplm/
-   Github: https://github.com/uber-research/PPLM
-   Openreview: https://openreview.net/forum?id=H1edEyBKDS

### XLNet: Generalized Autoregressive Pretraining for Language Understanding

-   Paper: https://arxiv.org/abs/1906.08237v1
-   Paperswithcode: https://paperswithcode.com/paper/xlnet-generalized-autoregressive-pretraining
-   Site: https://www.borealisai.com/en/blog/understanding-xlnet/
-   Site: https://mlexplained.com/2019/06/30/paper-dissected-xlnet-generalized-autoregressive-pretraining-for-language-understanding-explained/
-   Github: https://github.com/zihangdai/xlnet

### ALBERT: A Lite BERT for Self-supervised Learning of Language Representations

-   Paper: https://arxiv.org/abs/1909.11942
-   Paperswithcode: https://paperswithcode.com/paper/albert-a-lite-bert-for-self-supervised
-   Openreview: https://openreview.net/forum?id=H1eA7AEtvS

## LM applied to areas

### DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation

-   Paper: https://arxiv.org/abs/1911.00536
-   Github: https://github.com/microsoft/DialoGPT
-   ResearchGate: https://www.researchgate.net/publication/337019571_DialoGPT_Large-Scale_Generative_Pre-training_for_Conversational_Response_Generation

#### Hyperparameters

-   mostly unknown

#### Dataset

-   scraped from Reddit
    -   total of 1.8B words
    -   more info in Section 2
-   byte pair tokenization
-   train-val ratio
-   token portions
-   perplexity calculation

#### Model

-   based on gpt2
-   vocab size: 50,257

#### Notes

-   trained on 16 V100s
-   trained until no progress in validation sets
-   117M and 345M models trained for 5 epochs
-   1.5B for 3 epochs
-   All training data is compressed and lazy-loaded
-   dynamic batching

### DLGNet: A Transformer-based Model for Dialogue Response Generation

-   Paper: https://arxiv.org/abs/1908.01841
-   Paperswithcode: https://paperswithcode.com/paper/multi-turn-dialogue-response-generation-with

### A Pre-training Based Personalized Dialogue Generation Model with Persona-sparse Data

### Generating Sentiment-Preserving Fake Online Reviews Using Neural Language Models and Their Human- and Machine-based Detection

-   Paper: https://arxiv.org/abs/1907.09177
-   Paperswithcode: https://paperswithcode.com/paper/a-pre-training-based-personalized-dialogue

### Automated Speech Generation from UN General Assembly Statements: Mapping Risks in AI Generated Texts

-   Paper: https://arxiv.org/abs/1906.01946v1
-   Paperwithcode: https://paperswithcode.com/paper/automated-speech-generation-from-un-general

### Language models and Automated Essay Scoring

-   Paper: https://arxiv.org/abs/1909.09482v1
-   Paperwithcode: https://paperswithcode.com/paper/language-models-and-automated-essay-scoring

### Read, Attend and Comment: A Deep Architecture for Automatic News Comment Generation

-   Paper: https://paperswithcode.com/paper/read-attend-and-comment-a-deep-architecture
-   Paperswithcode: https://paperswithcode.com/paper/read-attend-and-comment-a-deep-architecture

### Generating More Interesting Responses in Neural Conversation Models with Distributional Constraints

-   Paper: https://arxiv.org/abs/1809.01215
-   Paperswithcode: https://paperswithcode.com/paper/generating-more-interesting-responses-in
-   ResearchGate: https://www.researchgate.net/publication/327465014_Generating_More_Interesting_Responses_in_Neural_Conversation_Models_with_Distributional_Constraints

### Towards Controllable Story Generation

-   Paper: https://www.aclweb.org/anthology/W18-1505/
-   Paperswithcode: https://paperswithcode.com/paper/towards-controllable-story-generation

### The Book of Endless History: Authorial Use of GPT2 for Interactive Storytelling

-   Paper: https://link.springer.com/chapter/10.1007/978-3-030-33894-7_47

### Transfer Learning from Transformers to Fake News Challenge Stance Detection (FNC-1) Task

-   Paper: https://arxiv.org/abs/1910.14353
-   Paperswithcode: https://paperswithcode.com/paper/transfer-learning-from-transformers-to-fake

### Zero-Shot Paraphrase Generation with Multilingual Language Models

-   Paper: https://arxiv.org/abs/1911.03597
-   Paperswithcode: https://paperswithcode.com/paper/zero-shot-paraphrase-generation-with
-   Researchgate: https://www.researchgate.net/publication/337184303_Zero-Shot_Paraphrase_Generation_with_Multilingual_Language_Models

## Training objectives

### Neural Text Generation with Unlikelihood Training

-   Arxiv: https://arxiv.org/abs/1908.04319
-   Paperswithcode: https://paperswithcode.com/paper/neural-text-generation-with-unlikelihood
-   OpenReview: https://openreview.net/forum?id=SJeYe0NtvH
-   Github: https://github.com/facebookresearch/unlikelihood_training

## Controlled generation/style tranfer

### XL-Editor: Post-editing Sentences with XLNet

-   Paper: https://arxiv.org/abs/1910.10479
-   Paperswithcode: https://paperswithcode.com/paper/xl-editor-post-editing-sentences-with-xlnet

### Controlling Output Length in Neural Encoder-Decoders

-   Paper: https://arxiv.org/abs/1609.09552
-   Paperswithcode: https://paperswithcode.com/paper/controlling-output-length-in-neural-encoder
-   ResearchGate: https://www.researchgate.net/publication/311990518_Controlling_Output_Length_in_Neural_Encoder-Decoders

### Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer

-   Paper: https://arxiv.org/abs/1804.06437
-   Paperswithcode: https://paperswithcode.com/paper/delete-retrieve-generate-a-simple-approach-to
-   Github: https://github.com/rpryzant/delete_retrieve_generate
-   ResearchGate: https://www.researchgate.net/publication/324599986_Delete_Retrieve_Generate_A_Simple_Approach_to_Sentiment_and_Style_Transfer

### Controlling Linguistic Style Aspects in Neural Language Generation

-   Paper: https://arxiv.org/abs/1707.02633
-   Paperswithcode: https://paperswithcode.com/search?q_meta=&q=Controlling+Linguistic+Style+Aspects+in+Neural+Language+Generation
-   Researchgate: https://www.researchgate.net/publication/318337084_Controlling_Linguistic_Style_Aspects_in_Neural_Language_Generation

### Controllable Text Generation

-   Researchgate: https://www.researchgate.net/publication/314237754_Controllable_Text_Generation

### Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation

-   Paper: https://arxiv.org/abs/1905.05621v3
-   Paperswithcode: https://paperswithcode.com/paper/190505621
-   ResearchGate: https://www.researchgate.net/publication/335786272_Style_Transformer_Unpaired_Text_Style_Transfer_without_Disentangled_Latent_Representation
-   Github: https://github.com/fastnlp/style-transformer

### Learning to Control the Fine-grained Sentiment for Story Ending Generation

-   Paper: https://www.aclweb.org/anthology/P19-1603/
-   Paperswithcode: https://paperswithcode.com/paper/learning-to-control-the-fine-grained
-   Researchgate: https://www.researchgate.net/publication/335778627_Learning_to_Control_the_Fine-grained_Sentiment_for_Story_Ending_Generation

### Multiple-Attribute Text Rewriting

-   Paper: https://arxiv.org/abs/1811.00552
-   Paperswithcode: https://paperswithcode.com/paper/multiple-attribute-text-rewriting
-   OpenReview: https://openreview.net/forum?id=H1g2NhC5KQ
-   Github: https://github.com/facebookresearch/MultipleAttributeTextRewriting

### Plan-And-Write: Towards Better Automatic Storytelling

-   Paper: https://arxiv.org/abs/1811.05701
-   Paperswithcode: https://paperswithcode.com/paper/plan-and-write-towards-better-automatic
-   Researchgate: https://www.researchgate.net/publication/335380574_Plan-and-Write_Towards_Better_Automatic_Storytelling

## Discrimination

### The Detection of Distributional Discrepancy for Text Generation

-   Paper: https://arxiv.org/abs/1910.04859
-   Paperswithcode: https://paperswithcode.com/paper/the-detection-of-distributional-discrepancy
-   Openreview: https://openreview.net/forum?id=SylurJHFPS
-   ResearchGate: https://www.researchgate.net/publication/336510599_The_Detection_of_Distributional_Discrepancy_for_Text_Generation

### Real or Fake? Learning to Discriminate Machine from Human Generated Text

-   Paper: https://arxiv.org/abs/1906.03351
-   Paperswithcode: https://paperswithcode.com/paper/real-or-fake-learning-to-discriminate-machine
-   Researchgate: https://www.researchgate.net/publication/333678943_Real_or_Fake_Learning_to_Discriminate_Machine_from_Human_Generated_Text

## Sampling

### The Curious Case of Neural Text Degeneration

-   Paper: https://arxiv.org/abs/1904.09751
-   Paperwithcode: https://paperswithcode.com/paper/the-curious-case-of-neural-text-degeneration-1
-   Openreview: https://openreview.net/forum?id=rygGQyrFvH
-   ResearchGate: https://www.researchgate.net/publication/332590110_The_Curious_Case_of_Neural_Text_Degeneration

#### Notes

-   Nucleus sampling

## Evaluation

### Dynamic Evaluation of Transformer Language Models

-   Paper: https://arxiv.org/abs/1904.08378
-   Paperswithcode: https://paperswithcode.com/search?q_meta=&q=Dynamic+Evaluation+of+Transformer+Language+Models
-   ResearchGate: https://www.researchgate.net/publication/332494222_Dynamic_Evaluation_of_Transformer_Language_Models

### How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation

-   Paper: https://arxiv.org/abs/1603.08023
-   Paperswithcode: https://paperswithcode.com/paper/how-not-to-evaluate-your-dialogue-system-an
-   OpenReview: https://openreview.net/forum?id=r1beMGfuZH
-   ResearchGate: https://www.researchgate.net/publication/319770336_How_NOT_To_Evaluate_Your_Dialogue_System_An_Empirical_Study_of_Unsupervised_Evaluation_Metrics_for_Dialogue_Response_Generation

## Other

### Neural Text Generation: Past, Present and Beyond

-   Paper: https://arxiv.org/abs/1803.07133
-   Paperswithcode: https://paperswithcode.com/paper/neural-text-generation-past-present-and
-   ResearchGate: https://www.researchgate.net/publication/323904602_Neural_Text_Generation_Past_Present_and_Beyond
