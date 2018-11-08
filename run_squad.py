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
"""Run BERT on SQuAD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import math
import logging
import os
import modeling
import optimization
import tokenization
import six
import tensorflow as tf

from distutils.util import strtobool

import chainer
from chainer import functions as F
from chainer import training
from chainer.training import extensions
import numpy as np
import progressbar

_logger = logging.getLogger(__name__)


def get_arguments():
    parser = argparse.ArgumentParser(description='Arxiv')

    # Required parameters
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
        '--output_dir', required=True,
        help="The output directory where the model checkpoints will be written.")
    parser.add_argument(
        '--gpu', '-g', type=int, default=0,
        help="The id of gpu device to be used [0-]. If -1 is given, cpu is used.")

    # Other parameters
    parser.add_argument(
        '--train_file',
        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument(
        '--predict_file',
        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument(
        '--do_lower_case', type=strtobool, default='True',
        help="Whether to lower case the input text. Should be True for uncased models and False for cased models.")
    parser.add_argument(
        '--max_seq_length', type=int, default=384,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument(
        '--doc_stride', type=int, default=128,
        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument(
        '--max_query_length', type=int, default=64,
        help="The maximum number of tokens for the question. Questions longer than this will be truncated to this length.")

    parser.add_argument(
        '--do_train', type=strtobool, default='False',
        help="Whether to run training.")
    parser.add_argument(
        '--do_predict', '--do_eval', type=strtobool, default='False',
        help="Whether to run eval on the dev set.")
    parser.add_argument(
        '--train_batch_size', type=int, default=32,
        help="Total batch size for training.")
    parser.add_argument(
        '--predict_batch_size', '--eval_batch_size', type=int, default=8,
        help="Total batch size for predictions.")
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
    parser.add_argument(
        '--n_best_size', type=int, default=20,
        help="The total number of n-best predictions to generate in the "
        "nbest_predictions.json output file.")
    parser.add_argument(
        '--max_answer_length', type=int, default=30,
        help="The maximum length of an answer that can be generated. This is needed "
        "because the start and end predictions are not conditioned on one another.")
    parser.add_argument(
        '--verbose_logging', type=strtobool, default='False',
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.")

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


class SquadExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None):
        self.unique_id = np.array([unique_id], 'i')  # shape changed
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = np.array(input_ids, 'i')
        self.input_mask = np.array(input_mask, 'i')
        self.segment_ids = np.array(segment_ids, 'i')
        if start_position is not None:
            self.start_position = np.array(
                [start_position], 'i')  # shape changed
        if end_position is not None:
            self.end_position = np.array(
                [end_position], 'i')  # shape changed


def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    with tf.gfile.Open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                if is_training:
                    if len(qa["answers"]) != 1:
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset +
                                                       answer_length - 1]
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    actual_text = " ".join(
                        doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = " ".join(
                        tokenization.whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        _logger.warning("Could not find answer: '%s' vs. '%s'",
                                        actual_text, cleaned_answer_text)
                        continue

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position)
                examples.append(example)
    return examples


class Converter(object):
    """Converts examples to features, and then batches and to_gpu."""

    def __init__(self, is_training):
        self.is_training = is_training

    def __call__(self, features, gpu):
        return self.make_batch(features, gpu)

    def make_batch(self, features, gpu):
        """Creates a concatenated batch from a list of data and to_gpu."""

        all_unique_ids = []
        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_start_positions = []
        all_end_positions = []

        for feature in features:
            all_unique_ids.append(feature.unique_id)
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_segment_ids.append(feature.segment_ids)
            if self.is_training:
                all_start_positions.append(feature.start_position)
                all_end_positions.append(feature.end_position)

        def stack_and_to_gpu(data_list):
            sdata = F.pad_sequence(data_list, length=None, padding=0).array
            return chainer.dataset.to_device(gpu, sdata)

        batch_unique_ids = stack_and_to_gpu(all_unique_ids).astype('i')
        batch_input_ids = stack_and_to_gpu(all_input_ids).astype('i')
        batch_input_mask = stack_and_to_gpu(all_input_mask).astype('f')
        batch_input_segment_ids = stack_and_to_gpu(all_segment_ids).astype('i')
        if self.is_training:
            batch_start_positions = stack_and_to_gpu(
                all_start_positions).astype('i')[:, 0]  # shape should be (batch_size, )
            batch_end_positions = stack_and_to_gpu(
                all_end_positions).astype('i')[:, 0]  # shape should be (batch_size, )
            return (batch_input_ids, batch_input_mask,
                    batch_input_segment_ids,
                    batch_start_positions, batch_end_positions)
        else:
            return (batch_input_ids, batch_input_mask,
                    batch_input_segment_ids,
                    batch_unique_ids)


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(
            progressbar.ProgressBar()(examples)):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = _improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                    example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(
                    tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                if (example.start_position < doc_start or
                    example.end_position < doc_start or
                        example.start_position > doc_end or example.end_position > doc_end):
                    continue

                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position))
            unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file):
    """Write final predictions to the json file."""
    _logger.info("Writing predictions to: %s" % (output_prediction_file))
    _logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(
            progressbar.ProgressBar()(all_examples)):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            # (1, )-array -> int
            feature.unique_id = feature.unique_id.tolist()[0]
            result = unique_id_to_result[feature.unique_id]

            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if FLAGS.verbose_logging:
            _logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if FLAGS.verbose_logging:
            _logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                         orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if FLAGS.verbose_logging:
            _logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if FLAGS.verbose_logging:
            _logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def evaluate(examples, iterator, model, converter, device,
             predict_func):
    all_results = []
    all_features = []
    total_iter = len(iterator.dataset) // iterator.batch_size + 1
    for prebatch in progressbar.ProgressBar(max_value=total_iter)(iterator):
        batch = converter(prebatch, device)
        features_list = prebatch
        # In `batch`, features is concatenated and to_gpu.
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            result = predict_func(*batch)
            for i in range(len(prebatch)):
                unique_id = int(result["unique_ids"][i])
                start_logits = [float(x) for x in result["start_logits"][i]]
                end_logits = [float(x) for x in result["end_logits"][i]]
                all_results.append(
                    RawResult(
                        unique_id=unique_id,
                        start_logits=start_logits,
                        end_logits=end_logits))
                all_features.append(features_list[i])

    output_prediction_file = os.path.join(
        FLAGS.output_dir, "predictions.json")
    output_nbest_file = os.path.join(
        FLAGS.output_dir, "nbest_predictions.json")
    write_predictions(examples, all_features, all_results,
                      FLAGS.n_best_size, FLAGS.max_answer_length,
                      FLAGS.do_lower_case, output_prediction_file,
                      output_nbest_file)


def main():
    if not FLAGS.do_train and not FLAGS.do_predict and not FLAGS.do_print_test:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    if FLAGS.do_train:
        if not FLAGS.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if FLAGS.do_predict:
        if not FLAGS.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = read_squad_examples(
            input_file=FLAGS.train_file, is_training=True)
        train_features = convert_examples_to_features(
            train_examples, tokenizer, FLAGS.max_seq_length,
            FLAGS.doc_stride, FLAGS.max_query_length, is_training=True)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    bert = modeling.BertModel(config=bert_config)
    model = modeling.BertSQuAD(bert)
    if FLAGS.do_train:
        # If training, load BERT parameters only.
        ignore_names = ['output/W', 'output/b']
    else:
        # If only do_predict, load all parameters.
        ignore_names = None
    chainer.serializers.load_npz(
        FLAGS.init_checkpoint, model,
        ignore_names=ignore_names)

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
            train_features, FLAGS.train_batch_size)
        converter = Converter(is_training=True)
        updater = training.updaters.StandardUpdater(
            train_iter, optimizer,
            converter=converter,
            device=FLAGS.gpu,
            loss_func=model.compute_loss)
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
            trigger=(100, 'iteration'))  # logging

        trainer.extend(extensions.snapshot_object(
            model, 'model_snapshot_iter_{.updater.iteration}.npz'),
            trigger=(num_train_steps // 2, 'iteration'))  # TODO
        trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
        trainer.extend(extensions.PrintReport(
            ['iteration', 'main/loss',
             'main/accuracy', 'elapsed_time', 'eta']))
        trainer.extend(extensions.ProgressBar(update_interval=10))

        trainer.run()

    if FLAGS.do_predict:
        eval_examples = read_squad_examples(
            input_file=FLAGS.predict_file, is_training=False)
        eval_features = convert_examples_to_features(
            eval_examples, tokenizer, FLAGS.max_seq_length,
            FLAGS.doc_stride, FLAGS.max_query_length, is_training=False)
        test_iter = chainer.iterators.SerialIterator(
            eval_features, FLAGS.predict_batch_size,
            repeat=False, shuffle=False)
        converter = Converter(is_training=False)

        print('Evaluating ...')
        evaluate(eval_examples, test_iter, model,
                 converter=converter, device=FLAGS.gpu,
                 predict_func=model.predict)
        print('Finished.')


if __name__ == "__main__":
    main()
