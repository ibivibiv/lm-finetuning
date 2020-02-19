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

from sample import sample
from transformers import GPT2LMHeadModel, CTRLLMHeadModel, GPT2TokenizerFast, CTRLTokenizer, AdamW, get_linear_schedule_with_warmup

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2TokenizerFast),
    'ctrl': (CTRLLMHeadModel, CTRLTokenizer)
}


class TextDataset(Dataset):
    def __init__(self, path, tokenizer, args):

        start = time.time()

        if os.path.isdir(path):
            self.batches = []
            for f in glob.glob(os.path.join(path, '*.txt')):
                self.batches += self._tokenize(f, tokenizer, args)

        else:
            self.batches = self._tokenize(path, tokenizer, args)

        end = time.time()

        print(f'Dataset created in {int(end - start)} seconds')
        print(f'Dataset length: {len(self.batches)}')

    def _tokenize(self, path, tokenizer, args):
        batches = []
        with open(path, encoding="utf-8") as handle:
            if args.line_by_line:
                text = [line for line in handle.read().splitlines() if (
                    len(line) > 0 and not line.isspace())]
            else:
                text = handle.read()

        if args.line_by_line:
            batches = tokenizer.batch_encode_plus(
                text, add_special_tokens=True, max_length=args.seq_len)["input_ids"]
        else:
            tokenized_text = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(text))

            for i in range(len(tokenized_text) // args.seq_len):
                batches.append(tokenizer.build_inputs_with_special_tokens(
                    tokenized_text[i * args.seq_len: (i + 1) * args.seq_len]))

        return batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return torch.tensor(self.batches[index])


def finetune(args):
    wandb.init(project="lm-finetuning")

    if args.save_dir == None:
        save_dir = wandb.run.dir

    model, tokenizer = MODEL_CLASSES[args.model_type]

    model = model.from_pretrained(args.checkpoint).to(args.device)
    tokenizer = tokenizer.from_pretrained(args.checkpoint)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    train_dataset = TextDataset(args.train_path, tokenizer, args)
    val_dataset = TextDataset(args.val_path, tokenizer, args)

    def collate(examples):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate)

    # if args.accelerator == 'TPU':
    # from: https://github.com/pytorch/xla/issues/1191
    # import torch_xla.distributed.parallel_loader as pl

    # def len_parallelloader(self):
    #     return len(self._loader._loader)
    # pl.PerDeviceLoader.__len__ = len_parallelloader

    # train_dataloader = pl.ParallelLoader(
    #     train_dataloader, [args.device]).per_device_loader(args.device)

    train_steps = int(len(train_dataloader) /
                      args.grad_steps * args.epochs)

    if args.optimizer == 'AdamW':
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    elif args.optimizer == 'SGD':
        optimizer = SGD(optimizer_grouped_parameters, lr=args.lr)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
        0.1 * train_steps), num_training_steps=train_steps)

    if os.path.exists(args.checkpoint):
        print('Loading optimizer and scheduler')

        optimizer.load_state_dict(torch.load(
            os.path.join(args.checkpoint, 'optimizer.pt')))
        scheduler.load_state_dict(torch.load(
            os.path.join(args.checkpoint, 'scheduler.pt')))

    if args.accelerator == 'GPU':
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O1", loss_scale="dynamic")
    elif args.accelerator == 'TPU':
        import torch_xla.core.xla_model as xm

    wandb.watch(model, log='parameters')

    gradients = {}

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    if os.path.exists(args.checkpoint):
        global_step = int(args.checkpoint.split('-')[-1].split('/')[0])

        epochs_trained = global_step // (len(train_dataloader) //
                                         args.grad_steps)
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // args.grad_steps) * args.grad_steps

    for epoch in range(epochs_trained, args.epochs):
        train_loss = 0
        val_loss = 0

        print(f"Epoch: {epoch}")

        model.train()
        for i, batch in tqdm(enumerate(train_dataloader), total=int(len(train_dataset) / args.batch_size)):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, labels = batch.to(args.device), batch.to(args.device)

            out = model(inputs, labels=labels)
            loss = out[0]

            loss = loss / args.grad_steps

            train_loss += loss.item()

            if args.accelerator == 'GPU':
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (i + 1) % args.grad_steps == 0:
                if args.accelerator == 'GPU':
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), 1)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                if args.accelerator == 'TPU':
                    xm.optimizer_step(optimizer, barrier=True)
                else:
                    optimizer.step()

                scheduler.step()

                if global_step % args.logging_steps == 0:
                    wandb.log({"train_loss": loss.item() * args.grad_steps,
                               "learning_rate": scheduler.get_lr()[0]}, step=global_step)

                    if global_step % args.hist_steps == 0:
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                try:
                                    gradients[f"gradients/{name}"] = wandb.Histogram(
                                        param.grad.detach().cpu().numpy())
                                except:
                                    pass

                    wandb.log(gradients, step=global_step)

                optimizer.zero_grad()

                global_step += 1

                # Must be in grad_accum block b/c if it is > 0, the model will get saved multiple times
                if global_step % args.save_steps == 0:
                    print(f'Saving model at global step: {global_step}')
                    checkpoint_dir = os.path.join(
                        save_dir, f'checkpoint-{global_step}')

                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)

                    model.save_pretrained(checkpoint_dir)
                    # tokenizer.save_pretrained(checkpoint_dir)
                    torch.save(optimizer.state_dict(), os.path.join(
                        checkpoint_dir, 'optimizer.pt'))
                    torch.save(scheduler.state_dict(), os.path.join(
                        checkpoint_dir, 'scheduler.pt'))

        model.eval()
        with torch.no_grad():
            for j, batch in tqdm(enumerate(val_dataloader), total=int(len(val_dataset) / args.batch_size)):
                inputs, labels = batch.to(args.device), batch.to(args.device)

                out = model(inputs, labels=labels)
                loss = out[0]

                val_loss += loss.item()

        train_loss /= (i + 1)
        val_loss /= (j + 1)

        train_loss *= args.grad_steps

        train_perplexity = torch.exp(torch.tensor(train_loss))
        val_perplexity = torch.exp(torch.tensor(val_loss))

        print('Sampling from model:\n')
        out = sample(" ", model, tokenizer, length=args.sample_len, temperature=args.temperature,
                     top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty, n_samples=args.n_samples)
        print('\n')

        wandb.log({"train_epoch_loss": train_loss, "train_epoch_perplexity": train_perplexity, 'val_epoch_loss': val_loss, 'val_epoch_perplexity': val_perplexity, "samples": wandb.Table(columns=['Epoch', 'Text'], data=[
            f'{epoch}', out])}, step=global_step)

        message = f'Finished epoch {epoch} | Train loss: {train_loss} | Train perplexity: {train_perplexity} | Val Loss: {val_loss} | Val Perplexity: {val_perplexity}'
        print(message)

    model.save_pretrained(save_dir)
    # tokenizer.save_pretrained(save_dir)
    torch.save(optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pt'))
    torch.save(scheduler.state_dict(), os.path.join(save_dir, 'scheduler.pt'))


# def tpu(index, train_dataset_path, val_dataset_path, save_dir, model_type, checkpoint, optimizer, lr, batch_size, gradient_accumulation_steps, epochs, accelerator, logging_steps, histogram_steps, save_steps, n_samples, sample_len, temperature, top_k, top_p, repetition_penalty, debug):
#     print(index)
#     finetune(train_dataset_path, val_dataset_path, save_dir, model_type, checkpoint, optimizer, lr, batch_size, gradient_accumulation_steps, epochs, accelerator,
#              logging_steps, histogram_steps, save_steps, n_samples, sample_len, temperature, top_k, top_p, repetition_penalty, debug)


# def main(train_dataset_path=None, val_dataset_path=None, save_dir=None, model_type='gpt2', checkpoint='distilgpt2', optimizer='AdamW', lr=5e-5, batch_size=4, gradient_accumulation_steps=1, epochs=1, accelerator='GPU', logging_steps=10, histogram_steps=100, save_steps=100, n_samples=1, sample_len=256, temperature=1, top_k=0, top_p=0, repetition_penalty=1, debug=False, n_cores=1):
#     if accelerator == 'CPU' or accelerator == 'GPU':
#         finetune(train_dataset_path, val_dataset_path, save_dir, model_type, checkpoint, optimizer, lr, batch_size, gradient_accumulation_steps, epochs, accelerator,
#                  logging_steps, histogram_steps, save_steps, n_samples, sample_len, temperature, top_k, top_p, repetition_penalty, debug)
#     else:
#         import torch_xla.core.xla_model as xm
#         import torch_xla.distributed.xla_multiprocessing as xmp

#         xmp.spawn(tpu, args=(train_dataset_path, val_dataset_path, save_dir, model_type, checkpoint, optimizer, lr, batch_size, gradient_accumulation_steps, epochs, accelerator, logging_steps,
#                              histogram_steps, save_steps, n_samples, sample_len, temperature, top_k, top_p, repetition_penalty, debug), nprocs=n_cores)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', default=None,
                        type=str, required=True)
    parser.add_argument('--val_path', default=None,
                        type=str, required=True)
    parser.add_argument('--save_dir', default=None,
                        type=str, required=False)
    parser.add_argument('--seq_len', default=256,
                        type=int, required=False)
    parser.add_argument('--line_by_line', default=False,
                        action="store_true", required=False)

    parser.add_argument('--model_type', default='gpt2', type=str)
    parser.add_argument('--checkpoint', default='distilgpt2', type=str)
    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=4, type=float)
    parser.add_argument('--grad_steps', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)

    parser.add_argument('--accelerator', default='GPU', type=str)
    parser.add_argument('--logging_steps', default=10, type=int)
    parser.add_argument('--hist_steps', default=100, type=int)
    parser.add_argument('--save_steps', default=100, type=int)

    parser.add_argument('--n_samples', default=1, type=int)
    parser.add_argument('--sample_len', default=256, type=int)
    parser.add_argument('--temperature', default=1, type=int)
    parser.add_argument('--top_k', default=1, type=int)
    parser.add_argument('--top_p', default=1, type=int)
    parser.add_argument('--repetition_penalty', default=1, type=int)

    parser.add_argument('--debug', default=False, action="store_true")

    parser.add_argument('--n_cores', default=1, type=int)

    args = parser.parse_args()

    if args.debug:
        import ptvsd
        ptvsd.enable_attach(address=('localhost', 5678),
                            redirect_output=True)
        ptvsd.wait_for_attach()
        breakpoint()

    if args.accelerator == 'TPU':
        import torch_xla.core.xla_model as xm

        args.device = xm.xla_device()
    elif args.accelerator == 'GPU':
        args.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device("cpu")

    finetune(args)


if __name__ == "__main__":
    main()
