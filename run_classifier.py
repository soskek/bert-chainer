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
"""BERT finetuning runner."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import logging
import os
import modeling
import optimization
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
    parser.add_argument(
        '--init_checkpoint', '--load_model_file', required=True,
        help="Initial checkpoint (usually from a pre-trained BERT model)."
        " The model array file path.")
    parser.add_argument(
        '--data_dir', required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument(
        '--bert_config_file', required=True,
        help="The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.")
    parser.add_argument(
        '--task_name', required=True,
        help="The name of the task to train.")
    parser.add_argument(
        '--vocab_file', required=True,
        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument(
        '--output_dir', required=True,
        help="The output directory where the model checkpoints will be written.")
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
        '--do_train', type=strtobool, default='False',
        help="Whether to run training.")
    parser.add_argument(
        '--do_eval', type=strtobool, default='False',
        help="Whether to run eval on the dev set.")
    parser.add_argument(
        '--train_batch_size', type=int, default=32,
        help="Total batch size for training.")
    parser.add_argument(
        '--eval_batch_size', type=int, default=8,
        help="Total batch size for eval.")
    parser.add_argument(
        '--learning_rate', type=float, default=5e-5,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        '--num_train_epochs', type=float, default=3.0,
        help="Total number of training epochs to perform.")
    parser.add_argument(
        '--warmup_proportion', type=float, default=0.1,
        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument(
        '--save_checkpoints_steps', type=int, default=1000,
        help="How often to save the model checkpoint.")
    parser.add_argument(
        '--iterations_per_loop', type=int, default=1000,
        help="How many steps to make in each estimator call.")
    # add
    parser.add_argument(
        '--do_print_test', type=strtobool, default='False',
        help="Whether to print some outputs on the partial dev set.")

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
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = np.array(input_ids, 'i')
        self.input_mask = np.array(input_mask, 'i')
        self.segment_ids = np.array(segment_ids, 'i')
        self.label_id = np.array([label_id], 'i')  # shape changed


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type,
                              tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[8])
            text_b = tokenization.convert_to_unicode(line[9])
            label = tokenization.convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Converter(object):
    """Converts examples to features, and then batches and to_gpu."""

    def __init__(self, label_list, max_seq_length, tokenizer):
        self.label_list = label_list
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.label_map = {}
        for (i, label) in enumerate(label_list):
            self.label_map[label] = i

    def __call__(self, examples, gpu):
        return self.convert_examples_to_features(examples, gpu)

    def convert_examples_to_features(self, examples, gpu):
        """Loads a data file into a list of `InputBatch`s.

        Args:
          examples: A list of examples (`InputExample`s).
          gpu: int. The gpu device id to be used. If -1, cpu is used.

        """
        max_seq_length = self.max_seq_length
        tokenizer = self.tokenizer
        label_map = self.label_map

        features = []
        for (ex_index, example) in enumerate(examples):
            # momoize
            if getattr(example, 'tokens_a', None):
                tokens_a = tokenizer.tokenize(example.text_a)
            else:
                tokens_a = tokenizer.tokenize(example.text_a)
                example.tokens_a = tokens_a

            tokens_b = None
            if example.text_b:
                # memoize
                if getattr(example, 'tokens_b', None):
                    tokens_b = tokenizer.tokenize(example.text_b)
                else:
                    tokens_b = tokenizer.tokenize(example.text_b)
                    example.tokens_b = tokens_b

            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[0:(max_seq_length - 2)]

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
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            if tokens_b:
                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            label_id = label_map[example.label]
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id))

        return self.make_batch(features, gpu)

    def make_batch(self, features, gpu):
        """Creates a concatenated batch from a list of data and to_gpu."""

        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_label_ids = []

        for feature in features:
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_segment_ids.append(feature.segment_ids)
            all_label_ids.append(feature.label_id)

        def stack_and_to_gpu(data_list):
            sdata = F.pad_sequence(
                data_list, length=None, padding=0).array
            return chainer.dataset.to_device(gpu, sdata)

        batch_input_ids = stack_and_to_gpu(all_input_ids).astype('i')
        batch_input_mask = stack_and_to_gpu(all_input_mask).astype('f')
        batch_input_segment_ids = stack_and_to_gpu(all_segment_ids).astype('i')
        batch_input_label_ids = stack_and_to_gpu(
            all_label_ids).astype('i')[:, 0]  # shape should be (batch_size, )
        return (batch_input_ids, batch_input_mask,
                batch_input_segment_ids, batch_input_label_ids)


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


def main():
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
    }

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_print_test:
        raise ValueError("At least one of `do_train` or `do_eval` "
                         "or `do_print_test` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    # TODO: use special Adam from "optimization.py"
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    bert = modeling.BertModel(config=bert_config)
    model = modeling.BertClassifier(bert, num_labels=len(label_list))
    chainer.serializers.load_npz(
        FLAGS.init_checkpoint, model,
        ignore_names=['output/W', 'output/b'])

    if FLAGS.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(FLAGS.gpu).use()
        model.to_gpu()

    if FLAGS.do_train:
        # Adam with weight decay only for 2D matrices
        optimizer = optimization.WeightDecayForMatrixAdam(
            alpha=1.,  # ignore alpha. instead, use eta as actual lr
            eps=1e-6, weight_decay_rate=0.01)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(1.))

        train_iter = chainer.iterators.SerialIterator(
            train_examples, FLAGS.train_batch_size)
        converter = Converter(
            label_list, FLAGS.max_seq_length, tokenizer)
        updater = training.updaters.StandardUpdater(
            train_iter, optimizer,
            converter=converter,
            device=FLAGS.gpu)
        trainer = training.Trainer(
            updater, (num_train_steps, 'iteration'), out=FLAGS.output_dir)

        # learning rate (eta) scheduling in Adam
        lr_decay_init = FLAGS.learning_rate * \
            (num_train_steps - num_warmup_steps) / num_train_steps
        trainer.extend(extensions.LinearShift(  # decay
            'eta', (lr_decay_init, 0.), (num_warmup_steps, num_train_steps)))
        trainer.extend(extensions.WarmupShift(  # warmup
            'eta', 0., num_warmup_steps, FLAGS.learning_rate))
        trainer.extend(extensions.observe_value(
            'eta', lambda trainer: trainer.updater.get_optimizer('main').eta),
            trigger=(50, 'iteration'))  # logging

        trainer.extend(extensions.snapshot_object(
            model, 'model_snapshot_iter_{.updater.iteration}.npz'),
            trigger=(num_train_steps, 'iteration'))
        trainer.extend(extensions.LogReport(
            trigger=(50, 'iteration')))
        trainer.extend(extensions.PrintReport(
            ['iteration', 'main/loss',
             'main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.ProgressBar(update_interval=10))

        trainer.run()

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        test_iter = chainer.iterators.SerialIterator(
            eval_examples, FLAGS.train_batch_size * 2,
            repeat=False, shuffle=False)
        converter = Converter(
            label_list, FLAGS.max_seq_length, tokenizer)
        evaluator = extensions.Evaluator(
            test_iter, model, converter=converter, device=FLAGS.gpu)
        results = evaluator()
        print(results)

    # if you wanna see some output arrays for debugging
    if FLAGS.do_print_test:
        short_eval_examples = processor.get_dev_examples(FLAGS.data_dir)[:3]
        short_eval_examples = short_eval_examples[:FLAGS.eval_batch_size]
        short_test_iter = chainer.iterators.SerialIterator(
            short_eval_examples, FLAGS.eval_batch_size,
            repeat=False, shuffle=False)
        converter = Converter(
            label_list, FLAGS.max_seq_length, tokenizer)
        evaluator = extensions.Evaluator(
            test_iter, model, converter=converter, device=FLAGS.gpu)

        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                data = short_test_iter.__next__()
                out = model.bert.get_pooled_output(
                    *converter(data, FLAGS.gpu)[:-1])
                print(out)
                print(out.shape)
            print(converter(data, -1))


if __name__ == "__main__":
    main()
