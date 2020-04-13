import os
import argparse
import time
import glob
import math

from typing import Callable
from functools import wraps

import numpy as np
from tqdm import tqdm

import wandb

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import GPT2LMHeadModel, CTRLLMHeadModel, GPT2TokenizerFast, CTRLTokenizer, AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from optimizers import Adafactor

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2TokenizerFast),
    'ctrl': (CTRLLMHeadModel, CTRLTokenizer)
}


# From: https://github.com/PyTorchLightning/pytorch-lightning/blob/3ad6169f187ea41aa1534a1d9a3b978d053dca2b/pytorch_lightning/loggers/base.py#L10
def rank_zero_only(fn: Callable):
    """Decorate a logger method to run it only on the process with rank 0.
    Args:
        fn: Function to decorate
    """

    @wraps(fn)
    def wrapped_fn(self, *args, **kwargs):
        if self.rank == 0:
            fn(self, *args, **kwargs)

    return wrapped_fn

# From: https://github.com/PyTorchLightning/pytorch-lightning/issues/1116#issuecomment-599244414


class WandbLogger(WandbLogger):
    @rank_zero_only
    def finalize(self, status: str = 'success') -> None:
        return 0


class TextDataset(torch.utils.data.Dataset):
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
            # efficient uses less memory by going line-by-line. Drawbacks: if len(line) isn't a multiple of seq_len, the remainder will be left
            if args.efficient or args.fast:
                for line in handle:
                    self.n_original_tokens += len(line.split(" "))
                    if len(line) > 0 and not line.isspace():
                        text.append(line)
            # Default way reads in entire file into memory
            else:
                temp = handle.read()
                text.append(temp)
                self.n_original_tokens += len(temp.strip().split(" "))

        # Fast way uses `batch_encode_plus`. Drawbacks: only the first seq_len chars get kept
        if args.fast:
            batches = tokenizer.batch_encode_plus(
                text, add_special_tokens=True, max_length=args.seq_len)["input_ids"]
        else:
            for l in tqdm(text):
                tokenized_text = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(l))

                if args.n_tokens > -1:
                    tokenized_text = tokenized_text[:args.n_tokens]

                if len(tokenized_text) < args.seq_len:
                    batches.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text))
                else:
                    for i in range(math.ceil(len(tokenized_text) / args.seq_len)):
                        batches.append(tokenizer.build_inputs_with_special_tokens(
                            tokenized_text[i * args.seq_len: (i + 1) * args.seq_len]))

                if args.n_batches > -1 and len(batches) >= args.n_batches:
                    break

        self.n_tokens += sum([len(batch) for batch in batches])

        return batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return torch.tensor(self.batches[index])


class LM(pl.LightningModule):
    def __init__(self, hparams):
        super(LM, self).__init__()

        self.hparams = hparams
        self.args = hparams

        model, tokenizer = MODEL_CLASSES[self.args.model_type]
        self.model = model.from_pretrained(self.args.model_name)
        self.tokenizer = tokenizer.from_pretrained(self.args.model_name)

        self.train_dataset = TextDataset(
            self.args.train_path, self.tokenizer, self.args)

        self.table_data = []

    def forward(self, inputs, labels):
        return self.model(inputs, labels=labels)

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, batch)[0]

        if self.args.disable_lr_schedule:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
        else:
            lr = self.scheduler.get_last_lr()[0]

        return {'loss': loss, "log": {"train_loss": loss, "learning_rate": lr}}

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch, batch)[0]

        return {'val_loss': loss}

    def validation_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_ppl = torch.exp(val_loss_mean)
        adjusted_val_loss = val_loss_mean * \
            ((self.val_dataset.n_tokens - 1) /
             (self.val_dataset.n_original_tokens - 1))
        adjusted_val_ppl = torch.exp(adjusted_val_loss)

        if self.args.accelerator != "TPU":
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

            prompt = torch.tensor(self.tokenizer.encode(
                "<|endoftext|> ")).unsqueeze(0).to(device)
            outputs = self.model.generate(input_ids=prompt, max_length=self.args.sample_len, temperature=self.args.temperature,
                                          top_k=self.args.top_k, top_p=self.args.top_p, repetition_penalty=self.args.repetition_penalty, num_return_sequences=1)
            outputs = self.tokenizer.decode(
                outputs[0].cpu().numpy(), skip_special_tokens=True)
            print("\nSampling:")
            print(outputs)
            print("\n")

            self.table_data.append([f'{self.trainer.current_epoch}', outputs])

        metrics = {'epoch': self.trainer.current_epoch, 'val_loss': val_loss_mean, 'val_ppl': val_ppl, 'adjusted_val_ppl': adjusted_val_ppl,
                   "log": {'epoch': self.trainer.current_epoch, 'val_loss': val_loss_mean, 'val_ppl': val_ppl, 'adjusted_val_ppl': adjusted_val_ppl, "samples": wandb.Table(columns=['Epoch', 'Text'], data=self.table_data)}}

        return metrics

    def test_step(self, batch, batch_idx):
        loss = self.forward(batch, batch)[0]

        return {'test_loss': loss}

    def test_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_ppl = torch.exp(test_loss_mean)
        adjusted_test_ppl = torch.exp(
            test_loss_mean * ((self.test_dataset.n_tokens - 1) / (self.test_dataset.n_original_tokens - 1)))

        if args.accelerator != "TPU":
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            prompt = torch.tensor(self.tokenizer.encode(
                "<|endoftext|> ")).unsqueeze(0).to(device)
            outputs = self.model.generate(input_ids=prompt, max_length=self.args.sample_len, temperature=self.args.temperature,
                                          top_k=self.args.top_k, top_p=self.args.top_p, repetition_penalty=self.args.repetition_penalty, num_return_sequences=1)
            outputs = self.tokenizer.decode(
                outputs[0].cpu().numpy(), skip_special_tokens=True)
            print("Sampling:")
            print(outputs)
            print("\n")

            self.table_data.append([f'{self.trainer.current_epoch}', outputs])

        metrics = {'epoch': self.trainer.current_epoch, 'test_epoch_loss': test_loss_mean,
                   'test_ppl': test_ppl, 'adjusted_test_ppl': adjusted_test_ppl, "log": {'epoch': self.trainer.current_epoch, 'test_epoch_loss': test_loss_mean, 'test_ppl': test_ppl, 'adjusted_test_ppl': adjusted_test_ppl, "samples": wandb.Table(columns=['Epoch', 'Text'], data=self.table_data)}}

        return metrics

    def configure_optimizers(self):
        if args.optimizer == 'AdamW':
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in self.model.named_parameters() if not any(
                    nd in n for nd in no_decay)], "weight_decay": 0.0},
                {"params": [p for n, p in self.model.named_parameters() if any(
                    nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]

            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=args.lr, eps=1e-8)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=args.lr)
        elif args.optimizer == 'Adafactor':
            optimizer = Adafactor(
                self.model.parameters(), lr=args.lr, beta1=args.momentum)

        if self.args.disable_lr_schedule:
            return optimizer
        else:
            n_accelerators = 1
            if self.args.n_tpu_cores != None:
                n_accelerators *= self.args.n_tpu_cores
            elif self.args.n_gpus != None:
                n_accelerators *= self.args.n_gpus

            train_steps = int(
                (len(self.train_dataset) / (self.args.batch_size * n_accelerators)) * self.args.epochs)

            self.scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(0.1 * train_steps), num_training_steps=train_steps)

            return [optimizer], [{'scheduler': self.scheduler, 'interval': 'step'}]

    def collate(self, examples):
        if self.tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def train_dataloader(self):
        if self.args.accelerator == "TPU":
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True
            )
        else:
            sampler = torch.utils.data.RandomSampler(self.train_dataset)

        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, num_workers=4, collate_fn=self.collate, sampler=sampler)

        return train_dataloader

    def val_dataloader(self):
        self.val_dataset = TextDataset(
            self.args.val_path, self.tokenizer, self.args)

        sampler = None
        if self.args.accelerator == "TPU":
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.val_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True
            )

        val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.args.batch_size, num_workers=4, collate_fn=self.collate, sampler=sampler, shuffle=False)

        return val_dataloader

    def test_dataloader(self):
        self.test_dataset = TextDataset(
            self.args.test_path, self.tokenizer, self.args)

        sampler = None
        if self.args.accelerator == "TPU":
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.test_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True
            )

        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.args.batch_size, num_workers=4, collate_fn=self.collate, sampler=sampler, shuffle=False)

        return test_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', default='./data/wikitext-2-raw/wiki.train.raw',
                        type=str, required=False)
    parser.add_argument('--val_path', default='./data/wikitext-2-raw/wiki.valid.raw',
                        type=str, required=False)
    parser.add_argument('--test_path', default='./data/wikitext-2-raw/wiki.test.raw',
                        type=str, required=False)

    parser.add_argument('--seq_len', default=256, type=int, required=False)
    parser.add_argument('--n_tokens', default=-1, type=int, required=False)
    parser.add_argument('--n_batches', default=-1, type=int, required=False)
    parser.add_argument('--fast', default=False,
                        action="store_true", required=False)
    parser.add_argument('--efficient', default=False,
                        action="store_true", required=False)

    parser.add_argument('--model_type', default='gpt2', type=str)
    parser.add_argument('--model_name', default='distilgpt2', type=str)
    parser.add_argument('--checkpoint', default=None, type=str)

    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--momentum', default=0.0, type=float)
    parser.add_argument('--disable_lr_schedule',
                        default=False, action='store_true')

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--grad_steps', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)

    parser.add_argument('--accelerator', default='GPU', type=str)
    parser.add_argument('--n_gpus', default=None, type=int)
    parser.add_argument('--n_tpu_cores', default=None, type=int)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--apex_mode', default='O1', type=str)

    parser.add_argument('--sample_len', default=256, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--top_k', default=None, type=int)
    parser.add_argument('--top_p', default=None, type=float)
    parser.add_argument('--repetition_penalty', default=None, type=float)

    parser.add_argument('--track_grad_norm', default=-1, type=int)

    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--debug_run', default=False, action="store_true")

    args = parser.parse_args()

    if args.accelerator == "TPU":
        args.global_batch_size = args.batch_size * args.n_tpu_cores

    if args.debug:
        import ptvsd
        ptvsd.enable_attach(address=('localhost', 5678),
                            redirect_output=True)
        ptvsd.wait_for_attach()
        breakpoint()

    if args.accelerator == 'TPU':
        import torch_xla.core.xla_model as xm

    wandb.login()
    experiment = wandb.init(project="lm-finetuning", reinit=True)
    wandb_logger = WandbLogger(experiment=experiment)
    wandb_logger.log_hyperparams(args)

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
    checkpoint_callback = ModelCheckpoint(wandb.run.dir, save_top_k=-1)

    model = LM(args)
    trainer = pl.Trainer(max_epochs=args.epochs, accumulate_grad_batches=args.grad_steps, gpus=args.n_gpus, num_tpu_cores=args.n_tpu_cores,
                         precision=args.precision, amp_level=args.apex_mode, resume_from_checkpoint=args.checkpoint, logger=wandb_logger, track_grad_norm=args.track_grad_norm, fast_dev_run=args.debug_run, early_stop_callback=early_stopping_callback, checkpoint_callback=checkpoint_callback, progress_bar_refresh_rate=1, num_sanity_val_steps=0, log_gpu_memory='all')

    trainer.fit(model)
