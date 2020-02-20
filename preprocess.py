import os
import glob
import argparse

from tqdm import tqdm
import pandas as pd
import numpy as np


def imdb(args):
    reviews = pd.read_csv(args.data_path).values[:, 0]

    train_size = int(len(reviews) * 0.9)

    train_text = ""
    for review in tqdm(reviews[:train_size], total=len(reviews[:train_size])):

        train_text += f"{review} \n\n"

    val_text = ""
    for review in tqdm(reviews[train_size:], total=len(reviews[train_size:])):

        val_text += f"{review} \n\n"

    with open(os.path.join(args.out_path, f'{args.dataset}-train.txt'), 'w') as f:
        f.write(train_text)

    with open(os.path.join(args.out_path, f'{args.dataset}-val.txt'), 'w') as f:
        f.write(val_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default=None, type=str, required=False)
    parser.add_argument('--data_path', default=None, type=str, required=False)
    parser.add_argument('--out_path', default=None, type=str, required=False)
    parser.add_argument('--train_size', default=0.9,
                        type=float, required=False)

    args = parser.parse_args()

    if args.dataset == "imdb":
        imdb(args)
