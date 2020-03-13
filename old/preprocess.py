import os
import glob
import argparse
import random

from tqdm import tqdm
import pandas as pd
import numpy as np


def imdb(args):
    reviews = pd.read_csv(args.data_path).values[:, 0]

    train_size = int(len(reviews) * args.train_size)

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


def cnn_daily_mail(args):
    cnn_files = glob.glob(os.path.join(args.data_path, 'cnn/stories/*.story'))
    daily_mail_files = glob.glob(os.path.join(
        args.data_path, 'dailymail/stories/*.story'))

    files = cnn_files + daily_mail_files
    random.shuffle(files)

    train_size = int(len(files) * args.train_size)

    train_text = ""
    val_text = ""

    for file in tqdm(files[:train_size], total=len(files[:train_size])):
        with open(os.path.join(file), 'r') as f:
            train_text += f"{f.read()} \n"

    for file in tqdm(files[train_size:], total=len(files[train_size:])):
        with open(os.path.join(file), 'r') as f:
            val_text += f"{f.read()} \n"

    with open(os.path.join(args.out_path, f'{args.dataset}-train.txt'), 'a') as f:
        f.write(train_text)

    with open(os.path.join(args.out_path, f'{args.dataset}-val.txt'), 'a') as f:
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
    elif args.dataset == "cnn-daily-mail":
        cnn_daily_mail(args)
