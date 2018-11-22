# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import codecs
import collections
import json
import logging
import re

import modeling
import tokenization
import tensorflow as tf


from distutils.util import strtobool

import chainer
from chainer import functions as F
from chainer import training
from chainer.training import extensions
import numpy as np

_logger = logging.getLogger(__name__)


def get_arguments():
    parser = argparse.ArgumentParser(description='Arxiv')

    # Required parameters
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--layers', default='-1,-2,-3,-4', required=True)
    parser.add_argument(
        '--init_checkpoint', '--load_model_file', required=True,
        help="Initial checkpoint (usually from a pre-trained BERT model)."
        " The model array file path.")
    parser.add_argument(
        '--bert_config_file', required=True,
        help="The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.")
    parser.add_argument(
        '--vocab_file', required=True,
        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument(
        '--gpu', '-g', type=int, default=0,
        help="The id of gpu device to be used [0-]. If -1 is given, cpu is used.")

    # Other parameters
    parser.add_argument(
        '--do_lower_case', type=strtobool, default='True',
        help="Whether to lower case the input text. Should be True for uncased models and False for cased models.")
    parser.add_argument(
        '--max_seq_length', type=int, default=128,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help="Batch size for predictions.")

    # These args are NOT used in this port.
    parser.add_argument('--use_tpu', type=strtobool, default='False')
    parser.add_argument('--tpu_name')
    parser.add_argument('--tpu_zone')
    parser.add_argument('--gcp_project')
    parser.add_argument('--master')
    parser.add_argument('--num_tpu_cores', type=int, default=8)

    args = parser.parse_args()
    return args


FLAGS = get_arguments()


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = np.array(input_ids, 'i')
        self.input_mask = np.array(input_mask, 'i')
        self.input_type_ids = np.array(input_type_ids, 'i')


def make_batch(features, gpu):
    """Creates a concatenated batch from a list of data and to_gpu."""

    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)

    def stack_and_to_gpu(data_list):
        sdata = F.pad_sequence(
            data_list, length=None, padding=0).array
        return chainer.dataset.to_device(gpu, sdata)

    batch_input_ids = stack_and_to_gpu(all_input_ids).astype('i')
    batch_input_mask = stack_and_to_gpu(all_input_mask).astype('f')
    batch_input_type_ids = stack_and_to_gpu(all_input_type_ids).astype('i')
    return {'input_ids': batch_input_ids,
            'input_mask': batch_input_mask,
            'input_type_ids': batch_input_type_ids, }


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    # with tf.gfile.GFile(input_file, "r") as reader:
    with open(input_file, "r") as reader:
        while True:
            line = tokenization.convert_to_unicode(reader.readline())
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples


def main():
    layer_indexes = [int(x) for x in FLAGS.layers.split(",")]
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    bert = modeling.BertModel(config=bert_config)
    model = modeling.BertExtracter(bert)
    ignore_names = ['output/W', 'output/b']
    chainer.serializers.load_npz(
        FLAGS.init_checkpoint, model,
        ignore_names=ignore_names)

    if FLAGS.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(FLAGS.gpu).use()
        model.to_gpu()

    examples = read_examples(input_file=FLAGS.input_file)
    features = convert_examples_to_features(
        examples, FLAGS.max_seq_length, tokenizer)
    iterator = chainer.iterators.SerialIterator(
        features, FLAGS.batch_size,
        repeat=False, shuffle=False)
    # converter = converter(is_training=False)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    # with codecs.getwriter("utf-8")(open(FLAGS.output_file, "w")) as writer:
    with open(FLAGS.output_file, "w") as writer:
        for prebatch in iterator:
            unique_ids = [f.unique_id for f in prebatch]
            batch = make_batch(prebatch, FLAGS.gpu)
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                encoder_layers = model.get_all_encoder_layers(
                    input_ids=batch['input_ids'],
                    input_mask=batch['input_mask'],
                    token_type_ids=batch['input_type_ids'])
            encoder_layers = [layer.array.tolist() for layer in encoder_layers]
            for b in range(len(prebatch)):  # batchsize loop
                unique_id = unique_ids[b]
                feature = unique_id_to_feature[unique_id]
                output_json = collections.OrderedDict()
                output_json["linex_index"] = unique_id
                all_features = []
                for (i, token) in enumerate(feature.tokens):
                    all_layers = []
                    for (j, layer_index) in enumerate(layer_indexes):
                        layer_output = encoder_layers[layer_index][b]
                        layers = collections.OrderedDict()
                        layers["index"] = layer_index
                        layers["values"] = [
                            round(float(x), 6) for x in layer_output[i]
                        ]
                        all_layers.append(layers)
                    features = collections.OrderedDict()
                    features["token"] = token
                    features["layers"] = all_layers
                    all_features.append(features)
                output_json["features"] = all_features
                writer.write(json.dumps(output_json) + "\n")


if __name__ == "__main__":
    main()
