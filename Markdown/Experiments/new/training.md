# training

# datasets

- wikitext2
- wikitext103
- imdb

# models

-   gpt2 - 124M parameters
-   gpt2-medium - 355M parameters
-   gpt2-large - 774M parameters
-   gpt2-xl - 1.5B parameters

# hyperparameters

| epochs | optimizer | learning rate |
| ------ | --------- | ------------- |
| 1      | Adafactor | 5e-5          |

# experiments

| dataset   | context len | model   | batch size | train loss | val loss | adj val loss | best epoch | run                  |
| --------- | ----------- |-------- | ---------- | ---------- | -------- | ------------ | ---------- | -------------------- |
| wikitext2 | 256         | gpt2    | 8          | 3.38       | 3.03     | 3.40         | 1          | winter-snow-1146     | 
| wikitext2 | 256 | gpt2 | 16 | 3.39 | 3.02 | 3.39 | exalted-water-1147 |
| wikitext2 | 512 | gpt2 | 8 | 3.29 | 2.94 | 3.31 | glamorous-lake-1148 |
| wikitext2 | 1024 | gpt2 | 8 | 3.21 | 2.87 | 3.22 | pleasant-dust-1149 |
| wikitext2 | 256 | gpt2-medium | 8 | 3.06 | 2.81 | 3.16 | leafy-plant-1150 |
| wikitext2 | 512 | gpt2-medium | 8 | 2.96 | 2.70 | 3.04 | electric-moon-1151 |
| wikitext2 | 1024 | gpt2-medium | 8 | 2.88 | 2.63 | 2.95 | volcanic-darkness-1152 |
| wikitext2 | 256 | gpt2-large | 8 | 2.92 | 2.74 | 3.08 |  dulcet-deluge-1153 |
| wikitext2 | 512 | gpt2-large | 8 | 2.79 | 2.59 | 2.91 | copper-wildflower-1154 |
| wikitext2 | 1024 | gpt2-large | 8 | 2.71 | 2.51 | 2.82 | twilight-plasma-1155 |
| wikitext2 | 256 | gpt2-xl | 8 | 2.83 | 2.71 | 3.04 | apricot-lake-1156 |
| wikitext2 | 512 | gpt2-xl | 8 | 2.69 | 2.52 | 2.87 | sweet-feather-1157 |
| wikitext2 | 1024 | gpt2-xl | 8 | 2.71 | 2.51 | 2.82 | pleasant-plasma-1159 |
| imdb | 256 | gpt2 | 8 | 4.09 (loss going up) | 4.13 | 5.32 | copper-silence-1160 |
| imdb | 256 | gpt2 | 16 | 3.83 (loss going up) | 3.77 | 4.85 | super-dream-1161 |
| imdb | 256 | gpt2 | 64 | 3.73 | 3.58 | 4.61 | floral-spaceship-1162 |
| imdb | 512 | gpt2 | 64 | 3.70 | 3.55 | 4.57 | lyric-cherry-1163 |
| imdb | 1024 | gpt2 | 64 | 3.68 | 3.52 | 4.54 | volcanic-frost-1164 |
| imdb | 256 | gpt2-medium | 64 | 3.45 | 3.35 | 4.31 | pretty-sun-1165 |
| imdb | 512 | gpt2-medium | 64 | 3.40 | 3.30 | 4.25 | twilight-microwave-1166 |
| imdb | 1024 | gpt2-medium | 64 | 3.38 | 3.30 | 4.26 | vital-snowball-1169 | 
