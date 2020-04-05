# Comparing Context Lengths

See how models trained on different context lengths aren't comparable. Train lms on some set context len, and evaluate on different context lens.

# Hyperparameters

-   gpt2-medium
-   trained on wikitext2
-   trained using a context len of 1024

# Results

| eval context len | loss | val_ppl | adj_val_ppl |
| ---------------- | ---- | ------- | ----------- |
| 1024             |
| 512              |
| 256              |
| 128              |
