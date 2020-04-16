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
    'gpt2': (TFGPT2LMHeadModel, GPT2TokenizerFast)
}


class TextDataset(object):
    def __init__(self, path, tokenizer, args):

        start = time.time()

        self.n_original_tokens = 0
        self.n_tokens = 0

        if os.path.isdir(path):
            self.batches = []
            self.labels = []
            for i, f in enumerate(glob.glob(os.path.join(path, '*.txt'))):
                batches, labels = self._tokenize(f, tokenizer, args, i)
                self.batches += batches
                self.labels += labels
        else:
            self.batches, self.labels = self._tokenize(
                path, tokenizer, args, 0)

        end = time.time()

        print(f'Dataset created in {int(end - start)} seconds')
        print(f'Dataset length: {len(self.batches)}')
        print(
            f'Num tokens: {self.n_tokens} | Num original tokens: {self.n_original_tokens}')

    def _tokenize(self, path, tokenizer, args, i):
        tokenized_control_code = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(args.control_codes[i]))

        batches = []
        labels = []

        text = []
        with open(path, encoding="utf-8") as handle:
            # efficient uses less memory by going line-by-line. Drawbacks: if len(line) isn't a multiple of seq_len, the remainder will be left
            if args.efficient:
                for line in handle:
                    self.n_original_tokens += len(line.split(" "))
                    if len(line) > 0 and not line.isspace():
                        text.append(wikitext_detokenizer(line))
            # Default way reads in entire file into memory
            else:
                temp = handle.read()
                self.n_original_tokens += len(temp.strip().split(" "))

                text.append(wikitext_detokenizer(temp))

        for l in tqdm(text):
            tokenized_text = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(l))

            if args.n_tokens > -1:
                tokenized_text = tokenized_text[:args.n_tokens]

            if len(tokenized_text) < args.seq_len - 1:
                if not args.min_seq_len:
                    example = tokenizer.build_inputs_with_special_tokens(
                        tokenized_control_code + tokenized_text)

                    batches.append(example[:-1])
                    labels.append(example[1:])
            else:
                for i in range(len(tokenized_text) // (args.seq_len - 1)):
                    example = tokenizer.build_inputs_with_special_tokens(
                        tokenized_control_code + tokenized_text[i * (args.seq_len - 1): (i + 1) * (args.seq_len - 1)])

                    batches.append(example[:-1])
                    labels.append(example[1:])

                    if args.n_batches > -1 and len(batches) >= args.n_batches:
                        break

        self.n_tokens += sum([len(batch) for batch in batches])

        return batches, labels


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(feature0, feature1):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'batches': _int64_feature(feature0),
        'labels': _int64_feature(feature1),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def get_dataset(args, tokenizer):

    if not args.use_serialized:
        train_dataset = TextDataset(args.train_path, tokenizer, args)
        val_dataset = TextDataset(args.val_path, tokenizer, args)

        train_len = int(len(train_dataset.batches) / args.batch_size)

        with tf.io.TFRecordWriter('train.tfrecords') as writer:
            for data, label in tqdm(zip(train_dataset.batches, train_dataset.labels)):
                example = serialize_example(data, label)
                writer.write(example)

        with tf.io.TFRecordWriter('val.tfrecords') as writer:
            for data, label in tqdm(zip(val_dataset.batches, val_dataset.labels)):
                example = serialize_example(data, label)
                writer.write(example)

    feature_description = {
        'batches': tf.io.FixedLenFeature((args.seq_len - 1), tf.int64),
        'labels': tf.io.FixedLenFeature((args.seq_len - 1), tf.int64),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        x = tf.io.parse_single_example(example_proto, feature_description)
        return (x['batches'], x['labels'])

    train_dataset = tf.data.TFRecordDataset(['temp.tfrecords'])
    train_dataset = train_dataset.map(_parse_function).shuffle(
        100).batch(args.batch_size, drop_remainder=True)

    val_dataset = tf.data.TFRecordDataset(['val.tfrecords'])
    val_dataset = val_dataset.map(_parse_function).shuffle(
        100).batch(args.batch_size, drop_remainder=True)

    # train_n_original_tokens = train_dataset.n_original_tokens
    # train_n_tokens = train_dataset.n_tokens

    # val_n_original_tokens = val_dataset.n_original_tokens
    # val_n_tokens = val_dataset.n_tokens

    # train_dataset = tf.data.Dataset.from_tensor_slices(
    #     (train_dataset.batches, train_dataset.labels))
    # train_dataset = train_dataset.shuffle(100).batch(
    #     args.batch_size, drop_remainder=True)

    # val_dataset = tf.data.Dataset.from_tensor_slices(
    #     (val_dataset.batches, val_dataset.labels))
    # val_dataset = val_dataset.shuffle(100).batch(
    #     args.batch_size, drop_remainder=True)

    return tokenizer, train_dataset, val_dataset, train_len


class AdjLoss(object):
    def __init__(self, n_tokens, n_original_tokens, name):
        self.n_tokens = n_tokens
        self.n_original_tokens = n_original_tokens

        self.__name__ = name

    def __call__(self, y_true, y_pred):
        cross_entropy = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True)

        cross_entropy *= ((self.n_tokens - 1) / (self.n_original_tokens - 1))

        return cross_entropy


class PPL(object):
    def __init__(self, name):
        self.__name__ = name

    def __call__(self, y_true, y_pred):
        cross_entropy = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True)

        ppl = math.e ** K.mean(cross_entropy)

        return ppl


class AdjPPL(object):
    def __init__(self, n_tokens, n_original_tokens, name):
        self.n_tokens = n_tokens
        self.n_original_tokens = n_original_tokens

        self.__name__ = name

    def __call__(self, y_true, y_pred):
        cross_entropy = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True)

        cross_entropy *= ((self.n_tokens - 1) / (self.n_original_tokens - 1))

        ppl = math.e ** K.mean(cross_entropy)

        return ppl


class Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, dir):
        super(Checkpoint, self).__init__()

        self.dir = dir

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

    parser.add_argument('--train_path', default='./data/wikitext-2/wiki.train.tokens',
                        type=str, required=False)
    parser.add_argument('--val_path', default='./data/wikitext-2/wiki.valid.tokens',
                        type=str, required=False)
    parser.add_argument('--use_serialized', default=False, action='store_true')
    parser.add_argument('--control_codes', nargs='+',
                        default=['<|endoftext|>'])

    parser.add_argument('--seq_len', default=256, type=int, required=False)
    parser.add_argument('--n_tokens', default=-1, type=int, required=False)
    parser.add_argument('--n_batches', default=-1, type=int, required=False)
    parser.add_argument('--efficient', default=False,
                        action="store_true", required=False)
    parser.add_argument('--min_seq_len', default=False, action='store_true')

    parser.add_argument('--model_type', default='gpt2', type=str)
    parser.add_argument('--model_name', default='distilgpt2', type=str)
    parser.add_argument('--checkpoint', default=None, type=str)

    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--momentum', default=0.0, type=float)
    parser.add_argument('--relative_update_scale',
                        default=False, action='store_true')
    parser.add_argument('--disable_lr_schedule',
                        default=False, action='store_true')

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--grad_steps', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)

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

    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    with strategy.scope():
        model, tokenizer = MODEL_CLASSES[args.model_type]
        model = model.from_pretrained(args.model_name)

        tokenizer = tokenizer.from_pretrained(args.model_name)

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

    tokenizer, train_dataset, val_dataset, train_len = get_dataset(
        args, tokenizer)

    ####
    # add args
    ####
    print(train_len)

    n_train_steps = train_len * args.epochs

    wandb_callback = WandbCallback(log_weights=True)
    checkpoint_callback = Checkpoint(wandb.run.dir)

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

    # outputs = model.generate(bos_token_id=tokenizer.bos_token_id, max_length=256, temperature=1.0,
    #                          top_k=None, top_p=None, repetition_penalty=None, num_return_sequences=1)
    # outputs = tokenizer.decode(
    #     outputs[0].cpu().numpy(), skip_special_tokens=True)
    # print(outputs)


if __name__ == "__main__":
    main()
