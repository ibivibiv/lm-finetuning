# Some code from: https://gist.github.com/LysandreJik/c958925768eb6a9a72609ea99561d1cb

import os
import glob
import time
from tqdm import tqdm

import tensorflow as tf
import tensorflow_datasets
from transformers import *

from tensorflow.keras.mixed_precision import experimental as mixed_precision


def finetune():
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)

    # loss_scale = policy.loss_scale
    # print('Loss scale: %s' % loss_scale)
    tf.config.optimizer.set_experimental_options(
        {"auto_mixed_precision": True})

    model = TFGPT2LMHeadModel.from_pretrained('distilgpt2')
    tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')

    with open('./data/wikitext-2-raw/wiki.train.raw', encoding="utf-8") as handle:
        text = handle.read()

    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    inputs = []
    labels = []

    if len(tokenized_text) < 256:
        example = tokenizer.build_inputs_with_special_tokens(tokenized_text)
        inputs.append(example[:-1])
        labels.append(example[1:])
    else:
        for i in range(len(tokenized_text) // 256):
            example = tokenizer.build_inputs_with_special_tokens(
                tokenized_text[i * 256: (i + 1) * 256])
            inputs.append(example[:-1])
            labels.append(example[1:])

    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset = dataset.shuffle(100).batch(16, drop_remainder=True)

    # dataset = dataset.map(lambda x: tf.cast(x, dtype=tf.float16))

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=3e-5, epsilon=1e-08)
    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
        optimizer, "dynamic")

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=[
                  loss, *[None] * model.config.n_layer], metrics=[metric])
    model.fit(dataset, epochs=3)


if __name__ == "__main__":
    # import ptvsd
    # ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
    # ptvsd.wait_for_attach()
    # breakpoint()

    finetune()
