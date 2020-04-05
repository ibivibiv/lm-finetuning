# DialoGPT

-   Paper: https://arxiv.org/abs/1911.00536
-   Github: https://github.com/microsoft/DialoGPT
-   ResearchGate: https://www.researchgate.net/publication/337019571_DialoGPT_Large-Scale_Generative_Pre-training_for_Conversational_Response_Generation

## Hyperparameters

-   lr schedule: Noam with warmup
-   epochs: 3-5

## Dataset

-   147M reddit comment chains
-   1.8B words

## Model

-   finetuned version of gpt2, gpt2-medium, and gpt2-large

## Notes

-   trained on 16 V100s
-   not much details
