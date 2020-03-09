# TPU

-   gpt2-xl

| cores | optimizer  | seqlen | precision | gpu memory usage |
| ----- | ---------- | ------ | --------- | ---------------- |
| 1     | AdamW      | 256    | 16        | ~6 GB            |
| 1     | AdamW      | 1024   | 16        | ~6 GB            |
| 1     | AdamW      | 1024   | 32        | oom              |
| 8     | SGD        | 256    | 16        | ~11GB            |
| 8     | AdamW      | 1024   | 16        | ~11GB            |
| 1     | SGD        | 256    | 32        | OOM              |
| 1     | SGD        | 1024   | 32        |                  |
| 1     | Adafacgtor | 1024   | 32        |                  |
