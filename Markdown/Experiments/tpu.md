# TPU

-   gpt2-xl

| cores | optimizer | seqlen | precision | TPU memory usage | platform |
| ----- | --------- | ------ | --------- | ---------------- | -------- |
| 1     | AdamW     | 256    | 16        | ~6 GB            | pt       |
| 1     | AdamW     | 1024   | 16        | ~6 GB            | pt       |
| 1     | AdamW     | 1024   | 32        | oom              | pt       |
| 8     | SGD       | 256    | 16        | ~11GB            | pt       |
| 8     | AdamW     | 1024   | 16        | ~11GB            | pt       |
| 1     | SGD       | 256    | 32        | OOM              | pt       |
| 8     | SGD       | 256    | 32        | ~13GB            | TF       |
| 8     | SGD       | 1024   | 32        | ~13GB            | TF       |
| 8     | Adafactor | 1024   | 32        | ~11GB            | TF       |
