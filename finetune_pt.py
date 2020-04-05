# Some code taken from Huggingface/transformers
import os
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

from optimizers import Adafactor

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2TokenizerFast),
    'ctrl': (CTRLLMHeadModel, CTRLTokenizer)
}


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
            if args.efficient or args.fast:
                for line in handle:
                    self.n_original_tokens += len(line.split(" "))
                    if len(line) > 0 and not line.isspace():
                        text.append(line)
            else:
                temp = handle.read()
                text.append(temp)
                self.n_original_tokens += len(temp.strip().split(" "))

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
                    for i in range(len(tokenized_text) // args.seq_len):
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


def sample(model, tokenizer, args):
    prompt = torch.tensor(tokenizer.encode(
        "<|endoftext|> ")).unsqueeze(0).to(args.device)
    outputs = model.generate(input_ids=prompt, max_length=args.sample_len, temperature=args.temperature,
                             top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty, num_return_sequences=1)
    outputs = tokenizer.decode(
        outputs[0].cpu().numpy(), skip_special_tokens=True)
    print("Sampling:")
    print(outputs)
    print("\n")


def run_eval(args):
    model, tokenizer = MODEL_CLASSES[args.model_type]

    model = model.from_pretrained(args.checkpoint).to(args.device)
    tokenizer = tokenizer.from_pretrained(args.checkpoint)

    val_dataset = TextDataset(args.val_path, tokenizer, args)

    def collate(examples):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate)

    val_loss = 0

    model.eval()
    with torch.no_grad():
        for j, batch in tqdm(enumerate(val_dataloader), total=int(len(val_dataset) / args.batch_size)):
            inputs, labels = batch.to(args.device), batch.to(args.device)

            out = model(inputs, labels=labels)
            loss = out[0]

            # check
            val_loss += loss.item()

    val_loss /= (j + 1)

    val_perplexity = torch.exp(torch.tensor(val_loss))
    adjusted_val_perplexity = torch.exp(torch.tensor(
        val_loss) * ((val_dataset.n_tokens - 1) / (val_dataset.n_original_tokens - 1)))

    sample(model, tokenizer, args)

    message = f'Loss: {round(val_loss, 4)} | Perplexity: {round(val_perplexity, 4)} | Adjusted Perplexity: {round(adjusted_val_perplexity, 4)}'
    print(message)


def finetune(args):
    wandb.init(project="lm-finetuning", config=args)

    if args.save_dir == None:
        args.save_dir = wandb.run.dir

    model, tokenizer = MODEL_CLASSES[args.model_type]

    model = model.from_pretrained(args.checkpoint).to(args.device)
    tokenizer = tokenizer.from_pretrained(args.checkpoint)

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

    train_steps = int(len(train_dataloader) /
                      args.grad_steps * args.epochs)

    if args.optimizer == 'AdamW':
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Adafactor':
        optimizer = Adafactor(
            model.parameters(), lr=args.lr, beta1=0)

    # check
    if args.lr_schedule:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
            0.1 * train_steps), num_training_steps=train_steps)

    if os.path.exists(args.checkpoint):
        print('Loading optimizer and scheduler')

        optimizer.load_state_dict(torch.load(
            os.path.join(args.checkpoint, 'optimizer.pt')))

        if args.lr_schedule:
            scheduler.load_state_dict(torch.load(
                os.path.join(args.checkpoint, 'scheduler.pt')))

    if args.accelerator == 'GPU' and args.fp16 == True:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.apex_mode, loss_scale="dynamic")
    elif args.accelerator == 'TPU':
        import torch_xla.core.xla_model as xm

    wandb.watch(model, log='parameters')

    gradients = {}
    table_data = []

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    if os.path.exists(args.checkpoint):
        global_step = int(args.checkpoint.split('-')[-1].split('/')[0])

        # check
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

            # check
            train_loss += loss.item()

            if args.accelerator == 'GPU' and args.fp16 == True:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (i + 1) % args.grad_steps == 0:
                # check
                if args.accelerator == 'GPU' and args.fp16 == True:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), 1)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                # check
                if args.accelerator == 'TPU':
                    xm.optimizer_step(optimizer, barrier=True)
                else:
                    optimizer.step()

                if args.lr_schedule:
                    scheduler.step()

                if global_step % args.logging_steps == 0:

                    if args.lr_schedule:
                        lr = optimizer.param_groups[0]['lr']
                    else:
                        lr = scheduler.get_lr()[0]

                    wandb.log({"train_loss": loss.item() * args.grad_steps,
                               "learning_rate": lr}, step=global_step)

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
                        args.save_dir, f'checkpoint-{global_step}')

                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)

                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    torch.save(optimizer.state_dict(), os.path.join(
                        checkpoint_dir, 'optimizer.pt'))

                    if args.lr_schedule:
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
        adjusted_train_perplexity = torch.exp(torch.tensor(
            train_loss) * ((train_dataset.n_tokens - 1) / (train_dataset.n_original_tokens - 1)))
        adjusted_val_perplexity = torch.exp(torch.tensor(
            val_loss) * ((val_dataset.n_tokens - 1) / (val_dataset.n_original_tokens - 1)))

        sample(model, tokenizer, args)

        table_data.append([f'{epoch}', out])
        wandb.log({"train_epoch_loss": train_loss, "train_perplexity": train_perplexity, "adjusted_train_perplexity": adjusted_train_perplexity, 'val_epoch_loss': val_loss,
                   'val_perplexity': val_perplexity, 'adjusted_val_perplexity': adjusted_val_perplexity, "samples": wandb.Table(columns=['Epoch', 'Text'], data=table_data)}, step=global_step)

        message = f'Finished epoch {epoch} | Train loss: {train_loss} | Train perplexity: {train_perplexity} | Adjusted Train perplexity: {adjusted_train_perplexity} | Val Loss: {val_loss} | Val Perplexity: {val_perplexity} | Adjusted Val Perplexity: {adjusted_val_perplexity}'
        print(message)

    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    torch.save(optimizer.state_dict(), os.path.join(
        args.save_dir, 'optimizer.pt'))

    if args.lr_schedule:
        torch.save(scheduler.state_dict(), os.path.join(
            args.save_dir, 'scheduler.pt'))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_path', default='./data/wikitext-2-raw/wiki.train.raw', type=str, required=False)
    parser.add_argument(
        '--val_path', default='./data/wikitext-2-raw/wiki.valid.raw', type=str, required=False)
    parser.add_argument('--save_dir', default=None,
                        type=str, required=False)

    parser.add_argument('--seq_len', default=256, type=int, required=False)
    parser.add_argument('--n_tokens', default=-1, type=int, required=False)
    parser.add_argument('--n_batches', default=-1, type=int, required=False)
    parser.add_argument('--fast', default=False,
                        action="store_true", required=False)
    parser.add_argument('--efficient', default=False,
                        action="store_true", required=False)

    parser.add_argument('--model_type', default='gpt2', type=str)
    parser.add_argument('--checkpoint', default='distilgpt2', type=str)

    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr_schedule', default=True, type=bool)

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--grad_steps', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)

    # check
    # add multiple tpu cores
    parser.add_argument('--accelerator', default='GPU', type=str)
    parser.add_argument('--fp16', default=False, action="store_true")
    parser.add_argument('--apex_mode', default='O1', type=str)

    parser.add_argument('--logging_steps', default=10, type=int)
    parser.add_argument('--hist_steps', default=100, type=int)
    parser.add_argument('--save_steps', default=100, type=int)

    parser.add_argument('--n_samples', default=1, type=int)
    parser.add_argument('--sample_len', default=256, type=int)
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument('--top_k', default=1, type=float)
    parser.add_argument('--top_p', default=1, type=float)
    parser.add_argument('--repetition_penalty', default=1, type=float)

    parser.add_argument('--eval_only', default=False, action="store_true")
    parser.add_argument('--debug', default=False, action="store_true")

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
    else:
        args.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    if args.eval_only:
        run_eval(args)
    else:
        finetune(args)


if __name__ == "__main__":
    main()
