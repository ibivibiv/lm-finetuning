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

from transformers import *

from detokenizer import wikitext_detokenizer

MODEL_CLASSES = {
    'gpt2': (TFGPT2LMHeadModel, GPT2TokenizerFast)
}


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(inputs, labels):
    feature = {
        'inputs': _int64_feature(inputs),
        'labels': _int64_feature(labels),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tokenize(i, paths, tokenizer, args):
    start = time.time()
    tokenized_control_code = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(args.control_codes[0]))

    n_examples = 0
    with tf.io.TFRecordWriter(os.path.join(args.save_path, f'{i}.tfrecord')) as writer:
        for path in tqdm(paths):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if len(line) > 0 and not line.isspace():
                        line = tokenizer.convert_tokens_to_ids(
                            tokenizer.tokenize(line))

                        if args.min_seq_len:
                            if len(line) < args.seq_len:
                                continue

                        for i in range(len(line) // (args.seq_len - 1)):
                            example = tokenizer.build_inputs_with_special_tokens(
                                tokenized_control_code + line[i * (args.seq_len - 1): (i + 1) * (args.seq_len - 1)])

                            inputs = example[:-1]
                            labels = example[1:]

                            example = serialize_example(inputs, labels)
                            writer.write(example)

                            if args.n_batches > -1 and len(n_examples) >= args.n_batches:
                                break

                            n_examples += 1

    end = time.time()
    print(f'#examples: {n_examples}')
    print(f'chunk processed in {int(end - start)} seconds')

    return n_examples


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', default='./data/wikitext-2/wiki.train.tokens',
                        type=str, required=False)
    parser.add_argument('--control_codes', nargs='+',
                        default=['<|endoftext|>'])
    parser.add_argument('--save_path', default='./', type=str, required=False)
    parser.add_argument('--files_per_tfrecord', default=1,
                        type=int, required=False)

    parser.add_argument('--seq_len', default=256, type=int, required=False)
    parser.add_argument('--n_tokens', default=-1, type=int, required=False)
    parser.add_argument('--n_batches', default=-1, type=int, required=False)
    parser.add_argument('--min_seq_len', default=False, action='store_true')

    parser.add_argument('--model_type', default='gpt2', type=str)
    parser.add_argument('--model_name', default='distilgpt2', type=str)

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

    _, tokenizer = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer.from_pretrained(args.model_name)

    start = time.time()

    n_examples = 0

    if os.path.isdir(args.path):
        files = glob.glob(os.path.join(args.path, '*'))
        print(f'Tokenizing {len(files)} files')

        for i in range(len(files) // args.files_per_tfrecord):
            files_subset = files[i *
                                 args.files_per_tfrecord: (i + 1) * args.files_per_tfrecord]

            n_examples += tokenize(i, files_subset, tokenizer, args)
    else:
        n_examples += tokenize(0, [args.path], tokenizer, args)

    end = time.time()

    print(f'Dataset created in {int(end - start)} seconds')
    print(f'#examples: {n_examples}')


if __name__ == "__main__":
    main()
