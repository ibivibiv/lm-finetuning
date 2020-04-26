# albert style lm

## notes

-   uses a factorized embedding between input one-hot encoded vectors and hidden dim
    -   default embedding parameter of 128
-   shares parameters for all layers
-   uses a sentence-order-prediction auxillary loss

## Todo

-   deal with multihead config params
-   make and upload config files
-   make conversion script
-   tf
    -   tf model
    -   tf pretrained models
-   pt
    -   pt model
    -   pt pretrained models
-   tokenizer
    -   fast?
-   optimizer
    -   tf and pt
