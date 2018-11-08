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
"""Common utility functions related to TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import six

import chainer
from chainer import functions as F
from chainer import links as L
import numpy as np


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        # GFile for (gs://) and (hdfs://)
        # with tf.gfile.GFile(json_file, "r") as reader:
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Linear3D(L.Linear):
    def __init__(self, *args, **kwargs):
        super(Linear3D, self).__init__(*args, **kwargs)

    def call(self, x):
        return super(Linear3D, self).__call__(x)

    def __call__(self, x):
        # TODO: efficient way
        if x.ndim == 2:
            return self.call(x)
        assert x.ndim == 3

        x_2d = F.concat(F.separate(x, axis=1), axis=0)
        out_2d = self.call(x_2d)
        out_3d = F.stack(F.split_axis(
            out_2d, x.shape[1], axis=0), axis=1)
        # (B, S, W)
        return out_3d


class LayerNormalization3D(L.LayerNormalization):
    def __init__(self, *args, **kwargs):
        super(LayerNormalization3D, self).__init__(*args, **kwargs)

    def call(self, x):
        return super(LayerNormalization3D, self).__call__(x)

    def __call__(self, x):
        # TODO: efficient way
        if x.ndim == 2:
            return self.call(x)
        assert x.ndim == 3

        x_2d = F.concat(F.separate(x, axis=1), axis=0)
        out_2d = self.call(x_2d)
        out_3d = F.stack(F.split_axis(
            out_2d, x.shape[1], axis=0), axis=1)
        # (B, S, W)
        return out_3d


class BertExtracter(chainer.Chain):
    # Useless wrapper of bert, only for de-serialization alignment
    # TODO: fix serialization of tf-BERT
    def __init__(self, bert):
        super(BertExtracter, self).__init__()
        with self.init_scope():
            self.bert = bert

    def get_pooled_output(self, *args, **kwargs):
        return self.bert.get_pooled_output(*args, **kwargs)

    def get_embedding_output(self, *args, **kwargs):
        return self.bert.get_embedding_output(*args, **kwargs)

    def get_all_encoder_layers(self, *args, **kwargs):
        return self.bert.get_all_encoder_layers(*args, **kwargs)

    def get_sequence_output(self, *args, **kwargs):
        return self.bert.get_sequence_output(*args, **kwargs)


class BertClassifier(chainer.Chain):
    def __init__(self, bert, num_labels):
        super(BertClassifier, self).__init__()
        with self.init_scope():
            self.bert = bert
            self.output = Linear3D(
                None, num_labels,
                initialW=create_initializer(initializer_range=0.02))

    def __call__(self, input_ids, input_mask, token_type_ids, labels):
        output_layer = self.bert.get_pooled_output(
            input_ids,
            input_mask,
            token_type_ids)
        output_layer = F.dropout(output_layer, 0.1)
        logits = self.output(output_layer)
        loss = F.softmax_cross_entropy(logits, labels)
        chainer.report({'loss': loss.array}, self)
        chainer.report({'accuracy': F.accuracy(logits, labels)}, self)
        return loss


# For showing SQuAD accuracy with heuristics
def check_answers(logits_var, labels_array, start_labels_array=None):
    if start_labels_array is not None:
        xp = chainer.cuda.get_array_module(labels_array)
        logits_array = logits_var.array.copy()
        assert (labels_array >= start_labels_array).all()
        # set -inf to score of positions before "start_labels"
        invalid_mask = xp.arange(logits_array.shape[1])[None] < \
            start_labels_array[:, None]
        logits_array = logits_array - invalid_mask * 10000.
        return (logits_array.argmax(axis=1) == labels_array)
    else:
        return (logits_var.array.argmax(axis=1) == labels_array)


class BertSQuAD(chainer.Chain):
    def __init__(self, bert):
        super(BertSQuAD, self).__init__()
        with self.init_scope():
            self.bert = bert
            self.output = Linear3D(
                None, 2,
                initialW=create_initializer(initializer_range=0.02))

    def __call__(self, input_ids, input_mask, token_type_ids):
        final_hidden = self.bert.get_sequence_output(
            input_ids,
            input_mask,
            token_type_ids)
        batch_size = final_hidden.shape[0]
        seq_length = final_hidden.shape[1]
        hidden_size = final_hidden.shape[2]

        final_hidden_matrix = F.reshape(
            final_hidden, [batch_size * seq_length, hidden_size])

        logits = self.output(final_hidden_matrix)

        logits = F.reshape(logits, [batch_size, seq_length, 2])
        logits = logits - (1 - input_mask[:, :, None]) * 1000.  # ignore pads
        logits = F.transpose(logits, [2, 0, 1])

        unstacked_logits = F.separate(logits, axis=0)

        (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])
        return (start_logits, end_logits)

    def compute_loss(self, input_ids, input_mask, token_type_ids,
                     start_positions, end_positions):
        (start_logits, end_logits) = self.__call__(
            input_ids, input_mask, token_type_ids)
        start_loss = F.softmax_cross_entropy(start_logits, start_positions)
        end_loss = F.softmax_cross_entropy(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2.0
        chainer.report({'loss': total_loss.array}, self)

        accuracy = (check_answers(start_logits, start_positions) *
                    check_answers(end_logits, end_positions, start_positions)).mean()
        chainer.report({'accuracy': accuracy}, self)
        return total_loss

    def predict(self, input_ids, input_mask, token_type_ids, unique_ids):
        (start_logits, end_logits) = self.__call__(
            input_ids, input_mask, token_type_ids)
        predictions = {
            "unique_ids": unique_ids[:, 0].tolist(),  # squeeze
            "start_logits": start_logits.array.tolist(),
            "end_logits": end_logits.array.tolist(),
        }
        return predictions


class BertModel(chainer.Chain):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:

    ```python
    # Already been converted into WordPiece token ids
    input_ids = array([[31, 51, 99], [15, 5, 0]])
    input_mask = array([[1, 1, 1], [1, 1, 0]])
    token_type_ids = array([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
      num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)
    model = modeling.BertModel(config=config)

    pooled_output = self.bert.get_pooled_output(
      input_ids,
      input_mask,
      token_type_ids)
    ...
    ```
    """

    def __init__(self,
                 config):
        """Constructor for BertModel.

        Args:
          config: `BertConfig` instance.

        Raises:
          ValueError: The config is invalid or one of the input tensor shapes
            is invalid.
        """
        super(BertModel, self).__init__()
        config = copy.deepcopy(config)
        self.dropout_prob = config.hidden_dropout_prob

        with self.init_scope():
            self.word_embeddings = L.EmbedID(
                config.vocab_size, config.hidden_size,
                initialW=create_initializer(config.initializer_range))
            self.token_type_embeddings = L.EmbedID(
                config.type_vocab_size, config.hidden_size,
                initialW=create_initializer(config.initializer_range))
            self.position_embeddings = L.EmbedID(
                config.max_position_embeddings, config.hidden_size,
                initialW=create_initializer(config.initializer_range))
            self.post_embedding_ln = LayerNormalization3D(config.hidden_size)

            self.encoder = Transformer(
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                intermediate_act_fn=get_activation(config.hidden_act),
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                initializer_range=config.initializer_range)
            self.pooler = Linear3D(
                config.hidden_size, config.hidden_size,
                initialW=create_initializer(config.initializer_range))

    def __call__(self,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 get_embedding_output=False,
                 get_all_encoder_layers=False,
                 get_sequence_output=False):
        """Encode by BertModel.

        Args:
          input_ids: int32 Tensor of shape [batch_size, seq_length].
          input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
          token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].

        Raises:
          ValueError: The config is invalid or one of the input tensor shapes
            is invalid.
        """
        input_shape = input_ids.shape
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = self.xp.ones(
                shape=[batch_size, seq_length], dtype=np.int32)

        if token_type_ids is None:
            token_type_ids = self.xp.zeros(
                shape=[batch_size, seq_length], dtype=np.int32)

        # Embed (sub-)words
        embedding_output = self.word_embeddings(input_ids)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        embedding_output = embedding_postprocessor(
            input_tensor=embedding_output,
            token_type_ids=token_type_ids,
            token_type_embedding=self.token_type_embeddings,
            position_embedding=self.position_embeddings,
            layer_norm=self.post_embedding_ln,
            dropout_prob=self.dropout_prob)
        if get_embedding_output:
            return embedding_output

        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        if get_all_encoder_layers:
            all_encoder_layers = self.encoder(
                input_tensor=embedding_output,
                attention_mask=attention_mask,
                do_return_all_layers=True)
            return all_encoder_layers
        else:
            sequence_output = self.encoder(
                input_tensor=embedding_output,
                attention_mask=attention_mask,
                do_return_all_layers=False)

        # The "pooler" converts the encoded sequence tensor of shape
        # [batch_size, seq_length, hidden_size] to a tensor of shape
        # [batch_size, hidden_size]. This is necessary for segment-level
        # (or segment-pair-level) classification tasks where we need a fixed
        # dimensional representation of the segment.
        if get_sequence_output:
            return sequence_output

        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        # (sosk) this "pooler" is not written in the paper.
        # first_token_tensor = F.squeeze(
        #    sequence_output[:, 0:1, :], axis=1)  # original
        first_token_tensor = sequence_output[:, 0]  # simplified
        pooled_output = F.tanh(self.pooler(first_token_tensor))
        return pooled_output

    def get_pooled_output(self, input_ids, input_mask=None, token_type_ids=None):
        return self.__call__(input_ids, input_mask, token_type_ids)

    def get_embedding_output(self, input_ids, input_mask=None, token_type_ids=None):
        return self.__call__(input_ids, input_mask, token_type_ids,
                             get_embedding_output=True)

    def get_all_encoder_layers(self, input_ids, input_mask=None, token_type_ids=None):
        return self.__call__(input_ids, input_mask, token_type_ids,
                             get_all_encoder_layers=True)

    def get_sequence_output(self, input_ids, input_mask=None, token_type_ids=None):
        return self.__call__(input_ids, input_mask, token_type_ids,
                             get_sequence_output=True)


def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
      input_tensor: float Tensor to perform activation.

    Returns:
      `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + F.erf(input_tensor / 2.0 ** 0.5))
    return input_tensor * cdf


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `F.relu`.

    Args:
      activation_string: String name of the activation function.

    Returns:
      A Python function corresponding to the activation function. If
      `activation_string` is None, empty, or "linear", this will return F.identity.
      If `activation_string` is not a string, it will return `activation_string`.

    Raises:
      ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return F.identity

    act = activation_string.lower()
    if act == "linear":
        return F.identity
    elif act == "relu":
        return F.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return F.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    # return tf.truncated_normal_initializer(stddev=initializer_range)
    # TODO: truncated_normal
    return chainer.initializers.Normal(initializer_range)


def embedding_postprocessor(input_tensor,
                            token_type_ids,
                            token_type_embedding,
                            position_embedding,
                            layer_norm,
                            dropout_prob=0.1):
    """Performs various post-processing on a word embedding tensor.

    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length,
        embedding_size].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
        Must be specified if `use_token_type` is True.
      token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
      token_type_embedding_name: string. The name of the embedding table variable
        for token type ids.
      position_embedding_name: string. The name of the embedding table variable
        for positional embeddings.
      initializer_range: float. Range of the weight initialization.
      max_position_embeddings: int. Maximum sequence length that might ever be
        used with this model. This can be longer than the sequence length of
        input_tensor, but cannot be shorter.
      dropout_prob: float. Dropout probability applied to the final output tensor.

    Returns:
      float tensor with same shape as `input_tensor`.

    Raises:
      ValueError: One of the tensor shapes or input values is invalid.
    """
    input_shape = input_tensor.shape
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]
    xp = token_type_embedding.xp

    max_position_embeddings = position_embedding.W.shape[0]
    if seq_length > max_position_embeddings:
        raise ValueError("The seq length (%d) cannot be greater than "
                         "`max_position_embeddings` (%d)" %
                         (seq_length, max_position_embeddings))

    output = input_tensor

    # if use_token_type: TODO? true/false
    if token_type_ids is None:
        raise ValueError("`token_type_ids` must be specified if"
                         "`use_token_type` is True.")
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    token_type_table = token_type_embedding.W
    flat_token_type_ids = F.reshape(token_type_ids, [-1])

    one_hot_ids = xp.zeros(
        (flat_token_type_ids.shape[0], token_type_table.shape[0])).astype('f')
    one_hot_ids[xp.arange(len(flat_token_type_ids)),
                flat_token_type_ids.array] = 1.
    token_type_embeddings = F.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = F.reshape(token_type_embeddings,
                                      [batch_size, seq_length, width])
    output += token_type_embeddings

    # Since the position embedding table is a learned variable, we create it
    # using a (long) sequence length `max_position_embeddings`. The actual
    # sequence length might be shorter than this, for faster training of
    # tasks that do not have long sequences.
    #
    # So `full_position_embeddings` is effectively an embedding table
    # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
    # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
    # perform a slice.

    # Only the last two dimensions are relevant (`seq_length` and `width`), so
    # we broadcast among the first dimensions, which is typically just
    # the batch size.

    position_embeddings = F.broadcast_to(
        position_embedding.W[None, :seq_length, :],
        output.shape)
    output += position_embeddings

    output = layer_norm(output)
    output = F.dropout(output, dropout_prob)

    return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
      from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    # numpy or cupy depending on GPU usage
    xp = chainer.cuda.get_array_module(from_tensor)

    from_shape = from_tensor.shape
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = to_mask.shape
    to_seq_length = to_shape[1]

    mask = xp.broadcast_to(
        to_mask[:, None],
        (batch_size, from_seq_length, to_seq_length))
    return mask


class AttentionLayer(chainer.Chain):
    def __init__(self,
                 num_attention_heads=1,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 attention_probs_dropout_prob=0.0,
                 initializer_range=0.02):
        """Performs multi-headed attention from `from_tensor` to `to_tensor`.

        This is an implementation of multi-headed attention based on "Attention
        is all you Need". If `from_tensor` and `to_tensor` are the same, then
        this is self-attention. Each timestep in `from_tensor` attends to the
        corresponding sequence in `to_tensor`, and returns a fixed-with vector.

        This function first projects `from_tensor` into a "query" tensor and
        `to_tensor` into "key" and "value" tensors. These are (effectively) a list
        of tensors of length `num_attention_heads`, where each tensor is of shape
        [batch_size, seq_length, size_per_head].

        Then, the query and key tensors are dot-producted and scaled. These are
        softmaxed to obtain attention probabilities. The value tensors are then
        interpolated by these probabilities, then concatenated back to a single
        tensor and returned.

        In practice, the multi-headed attention are done with transposes and
        reshapes rather than actual separate tensors.

        Args:
          num_attention_heads: int. Number of attention heads.
          size_per_head: int. Size of each attention head.
          query_act: (optional) Activation function for the query transform.
          key_act: (optional) Activation function for the key transform.
          value_act: (optional) Activation function for the value transform.
          attention_probs_dropout_prob:
          initializer_range: float. Range of the weight initializer.

        Raises:
          ValueError: Any of the arguments or tensor shapes are invalid.
        """
        super(AttentionLayer, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.query_act = query_act
        self.key_act = key_act
        self.value_act = value_act

        """
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`
        """
        with self.init_scope():
            # nobias?
            # `query_layer` = [B*F, N*H]
            self.query = L.Linear(
                None, num_attention_heads * size_per_head,
                initialW=create_initializer(initializer_range))
            # `key_layer` = [B*T, N*H]
            self.key = L.Linear(
                None, num_attention_heads * size_per_head,
                initialW=create_initializer(initializer_range))
            # `value_layer` = [B*T, N*H]
            self.value = L.Linear(
                None, num_attention_heads * size_per_head,
                initialW=create_initializer(initializer_range))

    def __call__(self, from_tensor, to_tensor,
                 attention_mask=None,
                 do_return_2d_tensor=False):
        """
        Args:
          from_tensor: float Tensor of shape [batch_size, from_seq_length, from_width].
          to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
          attention_mask: (optional) int32 Tensor of shape [batch_size,
            from_seq_length, to_seq_length]. The values should be 1 or 0. The
            attention scores will effectively be set to -infinity for any positions in
            the mask that are 0, and will be unchanged for positions that are 1.
          do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
            * from_seq_length, num_attention_heads * size_per_head]. If False, the
            output will be of shape [batch_size, from_seq_length, num_attention_heads
            * size_per_head].

        Returns:
          float Tensor of shape [batch_size, from_seq_length,
            num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
            true, this will be of shape [batch_size * from_seq_length,
            num_attention_heads * size_per_head]).
        """
        def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                                 seq_length, width):
            """
            output_tensor = F.stack(
                F.split_axis(input_tensor, num_attention_heads, axis=1),
                axis=1)
            # batch_size * seq_length, num_attention_heads, width

            output_tensor = F.stack(
                F.split_axis(output_tensor, seq_length, axis=0),
                axis=2)
            batch_size, num_attention_heads, seq_length, width
            """
            output_tensor = F.reshape(
                input_tensor,
                (batch_size, seq_length, num_attention_heads, width))
            output_tensor = F.transpose(output_tensor, [0, 2, 1, 3])
            return output_tensor

        from_shape = from_tensor.shape
        to_shape = to_tensor.shape

        if len(from_shape) != len(to_shape):
            raise ValueError(
                "The rank of `from_tensor` must match the rank of `to_tensor`.")

        if len(from_shape) == 3:
            batch_size = from_shape[0]
            from_seq_length = from_shape[1]
            to_seq_length = to_shape[1]
        elif len(from_shape) == 2:
            # TODO right?
            assert attention_mask is not None
            batch_size = attention_mask.shape[0]
            from_seq_length = attention_mask.shape[1]
            to_seq_length = attention_mask.shape[2]
            if (batch_size is None or from_seq_length is None or to_seq_length is None):
                raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values "
                    "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                    "must all be specified.")

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`

        from_tensor_2d = reshape_to_matrix(from_tensor)
        to_tensor_2d = reshape_to_matrix(to_tensor)

        # `query_layer` = [B*F, N*H]
        query_layer = self.query(from_tensor_2d)
        # `key_layer` = [B*T, N*H]
        key_layer = self.key(to_tensor_2d)
        # `value_layer` = [B*T, N*H]
        value_layer = self.value(to_tensor_2d)

        # `query_layer` = [B, N, F, H]
        query_layer = transpose_for_scores(
            query_layer, batch_size,
            self.num_attention_heads, from_seq_length, self.size_per_head)

        # `key_layer` = [B, N, T, H]
        key_layer = transpose_for_scores(
            key_layer, batch_size,
            self.num_attention_heads, to_seq_length, self.size_per_head)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = F.matmul(query_layer, key_layer, transb=True)
        attention_scores *= 1.0 / np.sqrt(self.size_per_head)

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            # attention_mask = F.expand_dims(attention_mask, axis=1)
            attention_mask = attention_mask[:, None]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
            adder = (1.0 - attention_mask) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += F.broadcast_to(adder, attention_scores.shape)

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        # (default softmax's axis is -1 in tf while 1 in chainer)
        # attention_probs = tf.nn.softmax(attention_scores)  # tf original
        attention_probs = F.softmax(attention_scores, axis=3)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = F.dropout(
            attention_probs, self.attention_probs_dropout_prob)

        # `value_layer` = [B, T, N, H]
        value_layer = F.reshape(
            value_layer,
            [batch_size, to_seq_length, self.num_attention_heads, self.size_per_head])

        # `value_layer` = [B, N, T, H]
        value_layer = F.transpose(value_layer, [0, 2, 1, 3])

        # `context_layer` = [B, N, F, H]
        context_layer = F.matmul(attention_probs, value_layer)  # right?

        # `context_layer` = [B, F, N, H]
        context_layer = F.transpose(context_layer, [0, 2, 1, 3])

        if do_return_2d_tensor:
            # `context_layer` = [B*F, N*V]
            context_layer = F.reshape(
                context_layer,
                [batch_size * from_seq_length, self.num_attention_heads * self.size_per_head])
            # right?
        else:
            # `context_layer` = [B, F, N*V]
            context_layer = F.reshape(
                context_layer,
                [batch_size, from_seq_length, self.num_attention_heads * self.size_per_head])

        return context_layer


class TransformerLayer(chainer.Chain):
    def __init__(self, hidden_size, intermediate_size,
                 num_attention_heads, attention_head_size,
                 attention_probs_dropout_prob, initializer_range):
        super(TransformerLayer, self).__init__()
        with self.init_scope():
            attention = AttentionLayer(
                num_attention_heads=num_attention_heads,
                size_per_head=attention_head_size,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                initializer_range=initializer_range)
            setattr(self, 'attention', attention)

            following_dense = Linear3D(
                None, hidden_size,
                initialW=create_initializer(initializer_range))
            following_ln = LayerNormalization3D(None)
            setattr(self, 'attention_output_dense', following_dense)
            setattr(self, 'attention_output_ln', following_ln)

            intermediate_dense1 = Linear3D(
                None, intermediate_size,
                initialW=create_initializer(initializer_range))
            intermediate_dense2 = Linear3D(
                None, hidden_size,
                initialW=create_initializer(initializer_range))
            intermediate_ln = LayerNormalization3D(None)
            setattr(self, 'intermediate_dense1', intermediate_dense1)
            setattr(self, 'intermediate_dense2', intermediate_dense2)
            setattr(self, 'intermediate_ln', intermediate_ln)


class Transformer(chainer.Chain):
    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 intermediate_act_fn=gelu,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02):
        """Multi-headed, multi-layer Transformer from "Attention is All You Need".

        This is almost an exact implementation of the original Transformer encoder.
        # Note that this (original) implementation of Transformer uses tricky reshapes
        # between 2D and 3D due to efficiency of computation.

        See the original paper:
        https://arxiv.org/abs/1706.03762

        Also see:
        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

        Args:
          hidden_size: int. Hidden size of the Transformer.
          num_hidden_layers: int. Number of layers (blocks) in the Transformer.
          num_attention_heads: int. Number of attention heads in the Transformer.
          intermediate_size: int. The size of the "intermediate" (a.k.a., feed
            forward) layer.
          intermediate_act_fn: function. The non-linear activation function to apply
            to the output of the intermediate/feed-forward layer.
          hidden_dropout_prob: float. Dropout probability for the hidden layers.
          attention_probs_dropout_prob: float. Dropout probability of the attention
            probabilities.
          initializer_range: float. Range of the initializer (stddev of truncated
            normal).

        Raises:
          ValueError: A Tensor shape or parameter is invalid.
        """
        super(Transformer, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

        attention_head_size = int(hidden_size / num_attention_heads)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dropout_prob = hidden_dropout_prob
        self.intermediate_act_fn = intermediate_act_fn

        with self.init_scope():
            for layer_idx in range(self.num_hidden_layers):
                layer_name = "layer_%d" % layer_idx
                layer = TransformerLayer(
                    hidden_size, intermediate_size,
                    num_attention_heads, attention_head_size,
                    attention_probs_dropout_prob, initializer_range)
                setattr(self, layer_name, layer)

    def __call__(self, input_tensor, attention_mask,
                 do_return_all_layers=False):
        """
        Args:
          input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
          attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
            seq_length], with 1 for positions that can be attended to and 0 in
            positions that should not be.
          do_return_all_layers: Whether to also return all layers or just the final
            layer.

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size], the final
          hidden layer of the Transformer.
          (If `do_return_all_layers` is true,
            this will be the list of the output of each layers.)
        """
        input_shape = input_tensor.shape

        prev_output = input_tensor
        all_layer_outputs = []
        for layer_idx in range(self.num_hidden_layers):
            layer_name = "layer_%d" % layer_idx
            layer = getattr(self, layer_name)
            layer_input = prev_output

            # Run attention
            attention_output = layer.attention(
                from_tensor=layer_input, to_tensor=layer_input,
                attention_mask=attention_mask,
                do_return_2d_tensor=True)

            # Run a linear projection of `hidden_size` then add a residual
            # with `layer_input`.
            attention_output = layer.attention_output_dense(attention_output)
            attention_output = F.dropout(
                attention_output, self.hidden_dropout_prob)

            layer_input = F.reshape(layer_input, attention_output.shape)
            attention_output = layer.attention_output_ln(
                attention_output + layer_input)  # residual

            intermediate_output = layer.intermediate_dense1(attention_output)
            # The activation is only applied to the "intermediate" hidden layer.
            intermediate_output = self.intermediate_act_fn(intermediate_output)
            # Down-project back to `hidden_size` then add the residual.
            layer_output = layer.intermediate_dense2(intermediate_output)
            layer_output = F.dropout(
                layer_output, self.hidden_dropout_prob)
            layer_output = layer.intermediate_ln(
                layer_output + attention_output)  # residual

            prev_output = layer_output
            all_layer_outputs.append(layer_output)

        if do_return_all_layers:
            final_outputs = []
            for layer_output in all_layer_outputs:
                final_output = reshape_from_matrix(layer_output, input_shape)
                final_outputs.append(final_output)
            return final_outputs
        else:
            final_output = reshape_from_matrix(prev_output, input_shape)
            return final_output

# Note that this (original) implementation of Transformer uses tricky reshapes
# between 2D and 3D due to efficiency of computation.


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor(i.e., a matrix)."""
    ndims = input_tensor.ndim
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = F.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    orig_dims = orig_shape_list[0:-1]
    width = output_tensor.shape[-1]
    return F.reshape(output_tensor, orig_dims + (width, ))
