import os
import re
import math
import glob
import time
import argparse
import pickle

import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops

import tensorflow_datasets as tfds

from transformers import *

import wandb
from wandb.keras import WandbCallback

from optimizers_tf import *
from detokenizer import wikitext_detokenizer

MODEL_CLASSES = {
    'gpt2': TFGPT2LMHeadModel
}


def get_dataset(args, tokenizer):
    feature_description = {
        'inputs': tf.io.FixedLenFeature((args.seq_len - 1), tf.int64),
        'labels': tf.io.FixedLenFeature((args.seq_len - 1), tf.int64),
    }

    def _parse_function(example_proto):
        x = tf.io.parse_single_example(example_proto, feature_description)
        return (x['inputs'], x['labels'])

    train_dataset = tf.data.TFRecordDataset(args.train_path)
    train_dataset = train_dataset.map(_parse_function).shuffle(
        100).batch(args.batch_size, drop_remainder=True)

    val_dataset = tf.data.TFRecordDataset(args.val_path)
    val_dataset = val_dataset.map(_parse_function).shuffle(
        100).batch(args.batch_size, drop_remainder=True)

    return tokenizer, train_dataset, val_dataset


class Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, dir, args):
        super(Checkpoint, self).__init__()

        self.dir = dir
        self.args = args

        self.n_batch = 0

    def on_batch_end(self, batch, logs=None):
        if (self.n_batch + 1) % self.args.save_batches == 0:
            checkpoint_dir = os.path.join(
                self.dir, f'checkpoint-batch-{self.n_batch}')

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            self.model.save_pretrained(checkpoint_dir)

        self.n_batch += 1

    def on_epoch_end(self, epoch, logs=None):
        checkpoint_dir = os.path.join(self.dir, f'checkpoint-{epoch}')

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.model.save_pretrained(checkpoint_dir)

    def on_train_end(self, logs=None):
        checkpoint_dir = os.path.join(self.dir, 'final_epoch')

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.model.save_pretrained(checkpoint_dir)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', nargs='*',
                        default=['gs://wikitext-2/0.tfrecord'], required=False)
    parser.add_argument('--val_path', nargs='*',
                        default=['gs://wikitext-2/0.tfrecord'], required=False)
    parser.add_argument('--train_len', default=100, type=int, required=False)

    parser.add_argument('--seq_len', default=256, type=int, required=False)
    parser.add_argument('--config_path', default='./', type=str)
    parser.add_argument('--model_type', default='gpt2', type=str)
    parser.add_argument('--tokenizer', default='gpt2', type=str)

    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--momentum', default=0.0, type=float)
    parser.add_argument('--relative_update_scale',
                        default=False, action='store_true')
    parser.add_argument('--disable_lr_schedule',
                        default=False, action='store_true')

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=1, type=int)

    parser.add_argument('--save_batches', default=1000, type=int)

    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    tf.random.set_seed(args.seed)

    if args.debug:
        import ptvsd
        ptvsd.enable_attach(address=('localhost', 5678),
                            redirect_output=True)
        ptvsd.wait_for_attach()
        breakpoint()

    wandb.login()
    wandb.init(project='lm-finetuning')

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)

    config = AutoConfig.from_pretrained(args.config_path)
    model = AutoModelWithLMHead.from_config(config=config)
    os.mkdir('./temp')
    model.save_pretrained('./temp')

    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    with strategy.scope():
        model = MODEL_CLASSES[args.model_type]
        model = model.from_pretrained('./temp', from_pt=True)

    os.rmdir('./temp')

    tokenizer = tokenizer.from_pretrained(args.tokenizer)
    # Can't use since TF models don't have resize_token_embeddings implemented
    # tokenizer.add_special_tokens(
    #     {'additional_special_tokens': args.control_codes})
    # model.resize_token_embeddings(len(tokenizer))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if args.optimizer == "SGD":
        optimizer = keras.optimizers.SGD(lr=args.lr, momentum=args.momentum)
    if args.optimizer == "AdamW":
        optimizer = AdamWeightDecay(learning_rate=args.lr)
    elif args.optimizer == "Adafactor":
        if args.relative_update_scale:
            optimizer = AdafactorOptimizer(
                beta1=args.momentum, multiply_by_parameter_scale=True)
        else:
            optimizer = AdafactorOptimizer(
                learning_rate=args.lr, beta1=args.momentum)

    tokenizer, train_dataset, val_dataset = get_dataset(
        args, tokenizer)

    n_train_steps = (args.train_len // args.batch_size) * args.epochs

    wandb_callback = WandbCallback()
    checkpoint_callback = Checkpoint(wandb.run.dir, args)

    model.compile(optimizer=optimizer, loss=[
                  loss, *[None] * model.config.n_layer])

    if args.disable_lr_schedule:
        model.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs, callbacks=[
                  wandb_callback, checkpoint_callback])
    else:
        lr_callback = WarmUpLinearDecayScheduler(
            learning_rate_base=args.lr, total_steps=n_train_steps, warmup_steps=int(0.1 * n_train_steps))

        model.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs, callbacks=[
                  wandb_callback, checkpoint_callback, lr_callback])


if __name__ == "__main__":
    main()
