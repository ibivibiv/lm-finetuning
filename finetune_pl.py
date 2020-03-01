import os
import fire
import pickle
import argparse
import time
import glob

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torch.optim import SGD

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

import wandb

from transformers import GPT2LMHeadModel, CTRLLMHeadModel, GPT2TokenizerFast, CTRLTokenizer, AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning import Trainer

import torch_xla.core.xla_model as xm


class TextDataset(Dataset):
    def __init__(self, path, tokenizer, args):

        start = time.time()

        self.n_original_tokens = 0
        self.n_tokens = 0

        if os.path.isdir(path):
            self.batches = []
            for f in glob.glob(os.path.join(path, '*.txt')):
                self.batches += self._tokenize(f, tokenizer, args)
        else:
            self.batches = self._tokenize(path, tokenizer, args)

        end = time.time()

        print(f'Dataset created in {int(end - start)} seconds')
        print(f'Dataset length: {len(self.batches)}')
        print(
            f'Num tokens: {self.n_tokens} | Num original tokens: {self.n_original_tokens}')

    def _tokenize(self, path, tokenizer, args):
        batches = []

        text = []
        with open(path, encoding="utf-8") as handle:
            temp = handle.read()
            text.append(temp)
            self.n_original_tokens += len(temp.strip().split(" "))

        for l in tqdm(text):
            tokenized_text = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(l))

            if len(tokenized_text) < 256:
                batches.append(
                    tokenizer.build_inputs_with_special_tokens(tokenized_text))
            else:
                for i in range(len(tokenized_text) // 256):
                    batches.append(tokenizer.build_inputs_with_special_tokens(
                        tokenized_text[i * 256: (i + 1) * 256]))

        self.n_tokens += sum([len(batch) for batch in batches])

        return batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return torch.tensor(self.batches[index])


class LM(pl.LightningModule):
    def __init__(self):
        super(LM, self).__init__()

        self.model = GPT2LMHeadModel.from_pretrained('distilgpt2')

    def forward(self, x):
        return self.model(x, labels=x)

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)[0]
        return {'loss': loss}

    def configure_optimizers(self):

        return torch.optim.SGD(self.parameters(), lr=1e-3)

    @pl.data_loader
    def train_dataloader(self):
        tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')

        train_dataset = TextDataset(
            './data/wikitext-2-raw/wiki.train.raw', tokenizer, None)

        def collate(examples):
            if tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, num_workers=4, collate_fn=collate, sampler=sampler)

        return train_dataloader


if __name__ == "__main__":
    model = LM()

    trainer = Trainer(
        num_tpu_cores=8, progress_bar_refresh_rate=1)
    trainer.fit(model)
