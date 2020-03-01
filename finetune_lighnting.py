# Some code taken from Huggingface/transformers
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

from optimizers import AdaFactor

import pytorch_lightning as pl


class LM(pl.LightningModule):
    def __init__(self):
        super(LM, self).__init__()

        self.lm = GPT2LMHeadModel.from_pretrained('distilgpt2')

    def forward(self, inputs):
        out = self.lm(inputs, labels=inputs)


if __name__ == "__main__":
